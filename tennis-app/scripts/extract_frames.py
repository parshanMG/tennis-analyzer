import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

def video_signature(p: Path) -> dict:
    st = p.stat()
    return {"src": str(p.resolve()), "src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns}

def load_meta(meta_path: Path) -> dict | None:
    try:
        if meta_path.exists():
            return json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        pass
    return None

def save_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def extract_one(video_path: Path, out_dir: Path, fps_target: float, ext: str, quality: int, max_w: int, overwrite: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_frames.meta.json"

    sig = video_signature(video_path)
    desired = {
        **sig,
        "fps_target": fps_target,
        "ext": ext,
        "quality": int(quality),
        "max_w": int(max_w) if max_w else 0,
    }

    # Skip if previously extracted with the same source and settings
    prev = load_meta(meta_path)
    if prev and not overwrite:
        equal = all(prev.get(k) == v for k, v in desired.items())
        if equal:
            print(f"Skip {video_path.name} (already extracted @ {fps_target} fps, ext={ext}). Use --overwrite to redo.")
            return 0
        else:
            print(f"Found existing frames for {video_path.name} but settings differ. Use --overwrite to regenerate.")
            return 0

    # If no meta but frames exist, also skip unless overwrite
    if not prev and not overwrite and any(out_dir.glob(f"*.{ext}")):
        print(f"Skip {video_path.name} (frames exist). Use --overwrite to redo.")
        return 0

    # Prefer FFmpeg first
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return 0

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, (max_w / max(w, 1))) if max_w else 1.0
    size = (int(round(w * scale)), int(round(h * scale)))

    print(f"[{video_path.name}] src_fps={fps:.2f} size={w}x{h} -> scale={scale:.3f} out={size[0]}x{size[1]} @ {fps_target} fps")

    interval = 1.0 / fps_target
    next_t = 0.0
    frame_idx = 0
    saved = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total_frames, unit="f", desc=f"Extract {video_path.stem}", leave=False)

    # Clear existing frames when overwriting
    if overwrite:
        for f in out_dir.glob(f"*.{ext}"):
            try: f.unlink()
            except Exception: pass

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        pbar.update(1)

        if scale < 1.0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        if t + (0.5 / fps) >= next_t:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            out_name = f"{video_path.stem}_{saved:06d}.{ext}"
            out_path = out_dir / out_name
            if ext.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            else:
                cv2.imwrite(str(out_path), frame)
            saved += 1
            next_t += interval

    pbar.close()
    cap.release()

    # Write/update metadata sentinel
    meta = {
        **desired,
        "frames": saved,
        "created": datetime.utcnow().isoformat() + "Z",
    }
    save_meta(meta_path, meta)
    print(f"Saved {saved} frames to {out_dir}")
    return saved

def main():
    ap = argparse.ArgumentParser(description="Extract frames from videos for labeling.")
    ap.add_argument("--videos", type=Path, default=Path("data/raw/videos"), help="Input videos folder")
    ap.add_argument("--out",    type=Path, default=Path("data/raw/frames"), help="Output frames root")
    ap.add_argument("--fps",    type=float, default=10.0, help="Target frame rate for sampling")
    ap.add_argument("--ext",    type=str, default="jpg", choices=["jpg", "png"], help="Image extension")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    ap.add_argument("--max-w",  type=int, default=1280, help="Max output width (0 to keep original)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    args = ap.parse_args()

    videos = sorted([p for p in args.videos.glob("*.*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}])
    if not videos:
        print(f"No videos found in {args.videos}")
        return

    total_saved = 0
    for vp in videos:
        out_dir = args.out / vp.stem
        total_saved += extract_one(vp, out_dir, args.fps, args.ext, args.quality, args.max_w, args.overwrite)

    print(f"Done. Total frames saved: {total_saved}")

if __name__ == "__main__":
    main()