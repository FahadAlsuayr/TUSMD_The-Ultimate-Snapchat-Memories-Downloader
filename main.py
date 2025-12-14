import argparse
import asyncio
import json
import re
import subprocess
import zipfile
import shutil
import threading
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Dependency Check
try:
    import httpx
    from PIL import Image
    from pydantic import BaseModel, Field, field_validator
    from tqdm.auto import tqdm 
    import ffmpeg
except ImportError as e:
    print(f"Critical Dependency Missing: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# ======================================================
# CONFIGURATION
# ======================================================

DOWNLOAD_SEM = asyncio.Semaphore(10) # Default safe limit
BAR_LOCK = threading.Lock()
FILE_LOCK = threading.Lock() 
FAILED_LOG = "failed_memories.json"
FFMPEG_TIMEOUT = 300 

ERROR_COUNT = 0 
USE_GPU = False 
PROCESSING_MODE = 1  # Default to Keep Both

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive"
}

# ======================================================
# DATA MODELS
# ======================================================

class Memory(BaseModel):
    date: datetime = Field(alias="Date")
    media_url: str | None = Field(default=None, alias="Media Download Url")
    download_link: str | None = Field(default=None, alias="Download Link")
    media_type: str | None = Field(default=None, alias="Media Type") 
    location: str = Field(default="", alias="Location")
    latitude: float | None = None
    longitude: float | None = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if not isinstance(v, str): return v
        v = v.strip()
        try: return datetime.fromisoformat(v)
        except ValueError: pass
        formats = ["%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S %Z"]
        for fmt in formats:
            try: return datetime.strptime(v, fmt)
            except ValueError: continue
        raise ValueError(f"Unknown date format: {v}")

    @property
    def filename(self):
        return self.date.strftime("%Y-%m-%d_%H-%M-%S")

# ======================================================
# UTILITIES
# ======================================================

def is_img(p): return p.suffix.lower() in (".jpg", ".jpeg", ".png")

def clean_debris(outdir: Path):
    """Removes temporary files from previous incomplete runs."""
    for p in outdir.glob("*_zip"):
        if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    for p in outdir.glob("*_TEMP"):
        if p.is_file(): p.unlink(missing_ok=True)
    for p in outdir.glob("*.zip"):
        if p.is_file(): p.unlink(missing_ok=True)

def log_failure(mem: Memory, error_msg: str):
    entry = mem.model_dump(by_alias=True)
    entry["Date"] = mem.date.strftime("%Y-%m-%d %H:%M:%S UTC")
    entry["_error"] = str(error_msg)

    with FILE_LOCK:
        data = []
        if os.path.exists(FAILED_LOG):
            try:
                with open(FAILED_LOG, "r", encoding="utf-8") as f: data = json.load(f)
            except: data = []
        
        # Avoid duplicate entries
        if not any(d.get("Media Download Url") == entry["Media Download Url"] for d in data):
            data.append(entry)
            
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def safe_write(path, data):
    try:
        path.write_bytes(data)
    except PermissionError:
        time.sleep(1)
        path.write_bytes(data)

def verify_file_integrity(path):
    """Checks for 0KB files and validates media headers."""
    if not path.exists() or path.stat().st_size == 0:
        raise Exception("File empty or missing")
    
    if is_img(path):
        try:
            with Image.open(path) as img: img.verify()
        except: raise Exception("Corrupt Image Header")
    else:
        try:
            ffmpeg.probe(str(path), cmd='ffprobe')
        except:
            if path.stat().st_size < 1024: raise Exception("Video too small/corrupt")

# ======================================================
# MEDIA PROCESSING
# ======================================================

def set_exif_data(path, mem: Memory):
    """Applies timestamp and GPS metadata to the file."""
    if not path.exists(): return
    
    try:
        mod_time = mem.date.timestamp()
        os.utime(path, (mod_time, mod_time))
    except: pass

    ts = mem.date.strftime("%Y:%m:%d %H:%M:%S")
    cmd = ["exiftool", "-overwrite_original", "-q", "-ignoreMinorErrors"]
    
    if is_img(path):
        cmd.extend([f"-DateTimeOriginal={ts}", f"-CreateDate={ts}", f"-ModifyDate={ts}"])
    else:
        cmd.extend([f"-CreateDate={ts}", f"-ModifyDate={ts}", f"-TrackCreateDate={ts}", f"-MediaCreateDate={ts}"])

    cmd.append(str(path))
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except: pass 

def sync_unzip(zip_path: Path, target_dir: Path):
    main, overlay = None, None
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.startswith("__MACOSX/"): continue
            if "-main." in name: main = Path(z.extract(name, target_dir))
            elif "-overlay." in name: overlay = Path(z.extract(name, target_dir))
    return main, overlay

def sync_merge_videos(main, overlay, out):
    """Merges video with overlay using FFmpeg. Falls back to CPU if GPU fails."""
    is_ovr_img = is_img(overlay)
    
    # Configure Hardware Acceleration
    input_args = {}
    output_args = {"vcodec": "libx264", "preset": "fast"} # CPU Default
    
    if USE_GPU:
        input_args = {"hwaccel": "cuda"}
        output_args = {"vcodec": "h264_nvenc", "preset": "p1"}

    input_main = ffmpeg.input(str(main), **input_args)
    if is_ovr_img: input_overlay = ffmpeg.input(str(overlay), loop=1)
    else: input_overlay = ffmpeg.input(str(overlay), **input_args)

    try:
        final = input_main.overlay(input_overlay, shortest=1)
        stream = ffmpeg.output(final, str(out), **output_args)
        cmd = ffmpeg.compile(stream, overwrite_output=True)
        subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise Exception(f"FFmpeg Merge Failed: {e}")

def sync_merge_images(main, overlay, out):
    verify_file_integrity(main)
    verify_file_integrity(overlay)
    base = Image.open(main).convert("RGBA")
    top = Image.open(overlay).convert("RGBA")
    if top.size != base.size:
        top = top.resize(base.size, Image.Resampling.LANCZOS)
    base.paste(top, (0, 0), top)
    base.convert("RGB").save(out, "JPEG")

# ======================================================
# CORE LOGIC
# ======================================================

async def fetch_binary(client, url):
    try:
        r = await client.get(url, timeout=45)
        return r.content if r.status_code == 200 else None
    except: return None

async def process_memory(client, mem, outdir, add_exif, specific_bar, total_bar, force_backup_link=False):
    if not mem.media_url and not mem.download_link: return
    name = mem.filename
    temp_path = outdir / f"{name}_TEMP"
    zip_path = outdir / f"{name}.zip"
    extract_dir = outdir / f"{name}_zip"
    
    # === FILE EXISTENCE CHECK (Zero-Trust) ===
    # Mode 3: Only check for MAIN.
    # Mode 1/2: Check for MERGED first, then MAIN.
    
    possible_files = []
    if PROCESSING_MODE == 3:
        possible_files = [outdir / f"{name}_MAIN.mp4", outdir / f"{name}_MAIN.jpg", outdir / f"{name}_MAIN.png"]
    else:
        possible_files = [
            outdir / f"{name}_MERGED.mp4", outdir / f"{name}_MERGED.jpg",
            outdir / f"{name}_MAIN.mp4", outdir / f"{name}_MAIN.jpg", outdir / f"{name}_MAIN.png"
        ]
    
    for f in possible_files:
        if f.exists():
            if f.stat().st_size == 0:
                f.unlink(missing_ok=True); continue
            try:
                verify_file_integrity(f)
                if specific_bar: specific_bar.update(1)
                if total_bar: total_bar.update(1)
                return # File is good, skip download
            except:
                f.unlink(missing_ok=True)

    # === DOWNLOAD LOGIC ===
    for attempt in range(1, 4):
        try:
            async with DOWNLOAD_SEM:
                links = [mem.media_url, mem.download_link] if not force_backup_link else [mem.download_link, mem.media_url]
                data = None
                for link in links:
                    if link:
                        data = await fetch_binary(client, link)
                        if data: break
                
                if not data: raise Exception("All download links failed")
                await asyncio.to_thread(safe_write, temp_path, data)

            is_zip = zipfile.is_zipfile(temp_path)

            if is_zip:
                temp_path.rename(zip_path)
                extract_dir.mkdir(exist_ok=True)
                main, overlay = await asyncio.to_thread(sync_unzip, zip_path, extract_dir)
                
                if not main: raise Exception("Empty Zip Archive")
                
                final_main = outdir / f"{name}_MAIN{main.suffix}"
                merged_out = outdir / f"{name}_MERGED{main.suffix}"

                # Save Main File
                await asyncio.to_thread(safe_write, final_main, main.read_bytes())
                if add_exif: await asyncio.to_thread(set_exif_data, final_main, mem)

                # Processing Logic based on Mode
                if PROCESSING_MODE == 3:
                    # Mode 3: Raw Only. Skip merge, delete zip/overlay implicitly via cleanup.
                    pass 
                else:
                    # Mode 1 & 2: Create Merge
                    if is_img(main) and is_img(overlay):
                        await asyncio.to_thread(sync_merge_images, final_main, overlay, merged_out)
                    else:
                        await asyncio.to_thread(sync_merge_videos, final_main, overlay, merged_out)
                    
                    if add_exif: await asyncio.to_thread(set_exif_data, merged_out, mem)

                    # Mode 2: Optimized (Delete Main if Merge Success)
                    if PROCESSING_MODE == 2 and merged_out.exists():
                        final_main.unlink(missing_ok=True)

            else:
                # Direct file (No overlay) - Same for all modes
                ext = ".mp4" if mem.media_type == "Video" else ".jpg"
                final_path = outdir / f"{name}_MAIN{ext}"
                temp_path.rename(final_path)
                await asyncio.to_thread(verify_file_integrity, final_path)
                if add_exif: await asyncio.to_thread(set_exif_data, final_path, mem)

            break # Success

        except Exception as e:
            if attempt == 3:
                log_failure(mem, str(e))
                with BAR_LOCK:
                    global ERROR_COUNT
                    ERROR_COUNT += 1
                    if total_bar: total_bar.set_postfix(errors=ERROR_COUNT)
            if attempt < 3: await asyncio.sleep(attempt * 2)
        finally:
            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)
            temp_path.unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)

    if specific_bar: specific_bar.update(1)
    if total_bar: total_bar.update(1)

# ======================================================
# EXECUTION
# ======================================================

async def run_batch(memories, outdir, add_exif, desc, n_workers, force_backup_link=False):
    global ERROR_COUNT
    ERROR_COUNT = 0
    
    img_queue = asyncio.Queue()
    vid_queue = asyncio.Queue()
    
    for m in memories:
        (img_queue if m.media_type == "Image" else vid_queue).put_nowait(m)

    print(f"\n🔹 {desc} | {len(memories)} items | Threads: {n_workers}")
    pbar_total = tqdm(total=len(memories), position=0, desc="TOTAL", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]")
    pbar_img = tqdm(total=img_queue.qsize(), position=1, desc="📸 IMG ", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    pbar_vid = tqdm(total=vid_queue.qsize(), position=2, desc="🎥 VID ", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    pbar_total.set_postfix(errors=0)

    async def worker(q, bar):
        async with httpx.AsyncClient(headers=HEADERS, timeout=60, follow_redirects=True) as client:
            while not q.empty():
                try:
                    mem = await q.get()
                    await process_memory(client, mem, outdir, add_exif, bar, pbar_total, force_backup_link)
                finally: q.task_done()

    tasks = []
    # Interleaved Launch
    for _ in range(n_workers):
        tasks.append(asyncio.create_task(worker(vid_queue, pbar_vid)))
        tasks.append(asyncio.create_task(worker(img_queue, pbar_img)))
        tasks.append(asyncio.create_task(worker(img_queue, pbar_img)))

    await asyncio.gather(img_queue.join(), vid_queue.join())
    for t in tasks: t.cancel()
    pbar_vid.close(); pbar_img.close(); pbar_total.close()

def scan_for_issues(all_memories, outdir, delete_bad=True):
    missing_or_bad = []
    for mem in tqdm(all_memories, desc="🔎 Verifying Files", unit="file"):
        base_name = mem.filename
        
        # Check based on mode preference logic
        candidates = []
        if PROCESSING_MODE == 3:
            candidates = [f"{base_name}_MAIN.mp4", f"{base_name}_MAIN.jpg", f"{base_name}_MAIN.png"]
        else:
            candidates = [f"{base_name}_MERGED.mp4", f"{base_name}_MERGED.jpg",
                          f"{base_name}_MAIN.mp4", f"{base_name}_MAIN.jpg", f"{base_name}_MAIN.png"]
        
        valid = False
        for c in candidates:
            fpath = outdir / c
            if fpath.exists() and fpath.stat().st_size > 0:
                valid = True; break
                
        if not valid: missing_or_bad.append(mem)
    return missing_or_bad

def get_user_mode():
    print("\n" + "="*50)
    print("   SNAPCHAT DOWNLOADER - MODE SELECTION")
    print("="*50)
    print("[1] Keep Both:   Saves original raw file AND merged video with overlay.")
    print("                 (Safest for archiving, uses most space)")
    print("[2] Optimized:   Keeps merged version if available. If no overlay, keeps Main.")
    print("                 (Best balance of usability and space)")
    print("[3] Raw Only:    Discards all overlays. Saves only the original raw media.")
    print("                 (Fastest, uses least space)")
    print("="*50)
    
    while True:
        choice = input("👉 Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

async def main():
    global USE_GPU
    global DOWNLOAD_SEM
    global PROCESSING_MODE
    
    # Interactive Menu
    PROCESSING_MODE = get_user_mode()

    ap = argparse.ArgumentParser(description="Snapchat Memory Downloader")
    ap.add_argument("json_file", nargs="?", default="json/memories_history.json")
    ap.add_argument("-o", "--output", default="./downloads")
    ap.add_argument("--no-exif", action="store_true", help="Disable EXIF/Timestamp fixing")
    ap.add_argument("-w", "--workers", type=int, default=0, help="Number of concurrent workers (0 = auto-detect)") 
    ap.add_argument("--gpu", action="store_true", help="Enable NVIDIA GPU acceleration (Default: Off)") 
    args = ap.parse_args()

    USE_GPU = args.gpu
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    clean_debris(outdir)

    # Hardware Detection & Config
    cpu_cores = os.cpu_count() or 4
    if args.workers == 0:
        workers_val = min(cpu_cores + 4, 20) # Conservative default
        print(f"⚙️ Auto-Config: Detected {cpu_cores} cores. Using {workers_val} workers.")
    else:
        workers_val = args.workers
        print(f"⚙️ Manual Config: Using {workers_val} workers.")

    DOWNLOAD_SEM = asyncio.Semaphore(workers_val + 20)
    
    print("\n🚀 CONFIGURATION:")
    print(f"   Mode:    {['Unknown', 'Keep Both', 'Optimized', 'Raw Only'][PROCESSING_MODE]}")
    print(f"   Accel:   {'GPU (NVENC)' if USE_GPU else 'CPU (Software)'}")
    print(f"   Workers: {workers_val}")

    try:
        raw = json.load(open(args.json_file, "r", encoding="utf-8"))
        all_memories = [Memory(**m) for m in raw["Saved Media"]]
    except Exception as e:
        print(f"❌ Error loading JSON: {e}"); return

    # Execution Phases
    await run_batch(all_memories, outdir, not args.no_exif, "Phase 1: Main Download", workers_val)
    
    print("\n🔎 Validating Downloads...")
    bad_memories = await asyncio.to_thread(scan_for_issues, all_memories, outdir)
    
    if bad_memories:
        print(f"⚠️ {len(bad_memories)} files missing/corrupt. Retrying...")
        await run_batch(bad_memories, outdir, not args.no_exif, "Phase 2: Repair Run", workers_val, force_backup_link=True)
    
    print("\n✅ Operation Complete.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())