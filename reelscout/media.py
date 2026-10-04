"""ffmpeg / OpenCV helpers: audio extraction, semantic segmentation, and frame sampling."""
from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Segment:
    """A semantic video segment with natural boundaries."""
    index: int
    start_time: float
    end_time: float
    duration: float
    boundary_type: str  # "scene_change", "silence", "topic_shift", "start", "end"


def video_key(path: Path) -> str:
    """Cheap content fingerprint so cached stages survive renames."""
    h = hashlib.sha1(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    info = dict(fps=fps, width=int(cap.get(3)), height=int(cap.get(4)), duration=n / fps if fps else 0)
    cap.release()
    return info


def extract_audio(video: Path, wav: Path) -> bool:
    wav.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        capture_output=True,
    )
    return r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000


def get_silence_intervals(video: Path, min_silence: float = 0.5, threshold_db: int = -30) -> List[float]:
    """Detect silence intervals using ffmpeg silencedetect. Returns list of silence start times."""
    r = subprocess.run([
        "ffmpeg", "-i", str(video), "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_silence}",
        "-f", "null", "-"
    ], capture_output=True, text=True, timeout=30)
    
    silence_starts = []
    for line in r.stderr.split("\n"):
        if "silence_start:" in line:
            try:
                t = float(line.split("silence_start:")[1].split()[0])
                silence_starts.append(t)
            except (IndexError, ValueError):
                pass
    return silence_starts


def detect_scene_changes(video: Path, sample_hz: float = 2.0, diff_thresh: float = 6.0) -> List[float]:
    """Detect scene changes using change-aware sampling. Returns list of scene change times."""
    cap = cv2.VideoCapture(str(Path(video)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_hz)))
    changes = []
    last_small = None
    idx = 0
    
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                diff = 999.0 if last_small is None else float(np.abs(small - last_small).mean())
                if last_small is not None and diff > diff_thresh:
                    changes.append(idx / fps)
                last_small = small
        idx += 1
    cap.release()
    return changes


def get_semantic_segments(video: Path, min_segment: float = 5.0, max_segment: float = 120.0) -> List[Segment]:
    """Segment video at natural boundaries: scene changes, silence, topic shifts."""
    info = probe(video)
    duration = info.get("duration", 0)
    if duration <= min_segment:
        return [Segment(0, 0.0, duration, duration, "single")]
    
    # Detect boundaries
    scene_changes = detect_scene_changes(video)
    silences = get_silence_intervals(video)
    
    # Combine and sort all boundaries
    boundaries = [0.0]
    boundaries.extend(scene_changes)
    boundaries.extend(silences)
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    
    # Merge too-close boundaries, enforce min/max segment duration
    segments = []
    seg_start = 0.0
    for i, b in enumerate(boundaries[1:], 1):
        seg_duration = b - seg_start
        if seg_duration >= min_segment or b == duration:
            if seg_duration > max_segment:
                # Split long segments
                num_splits = int(np.ceil(seg_duration / max_segment))
                split_dur = seg_duration / num_splits
                for j in range(num_splits):
                    s = seg_start + j * split_dur
                    e = min(s + split_dur, b)
                    segments.append(Segment(
                        len(segments), s, e, e - s,
                        "split" if j > 0 else ("scene_change" if b in [c for c in scene_changes] else "silence")
                    ))
            else:
                boundary_type = "scene_change" if b in [c for c in scene_changes] else "silence"
                segments.append(Segment(len(segments), seg_start, b, seg_duration, boundary_type))
            seg_start = b
    
    # Renumber
    for i, s in enumerate(segments):
        s.index = i
    
    return segments


def sample_frames(video: Path, out_dir: Path, max_frames: int = 16, sample_hz: float = 2.0,
                  diff_thresh: float = 6.0, max_side: int = 1280, fixed_hz: float = 1.0) -> list[dict]:
    """Sample frames from video: hybrid of change-aware + fixed-interval for production accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kept = []

    # Pass 1: Fixed interval sampling (guarantees uniform coverage)
    fixed_step = max(1, int(round(fps / fixed_hz))) if fixed_hz > 0 else 0
    fixed_kept = []
    idx = 0
    while True:
        if not cap.grab():
            break
        if fixed_step > 0 and idx % fixed_step == 0:
            ok, frame = cap.retrieve()
            if ok:
                fixed_kept.append((idx / fps, frame))
        idx += 1

    # Pass 2: Change-aware sampling (captures scene transitions)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    change_step = max(1, int(round(fps / sample_hz)))
    last_small = None
    idx = 0
    change_kept = []
    while True:
        if not cap.grab():
            break
        if idx % change_step == 0:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                diff = 999.0 if last_small is None else float(np.abs(small - last_small).mean())
                if diff > diff_thresh:
                    change_kept.append((idx / fps, frame))
                    last_small = small
        idx += 1
    cap.release()

    # Merge: prioritize change frames, fill gaps with fixed frames, deduplicate by time
    all_frames = []
    seen_times = set()
    
    # Add change frames first (highest priority)
    for t, frame in change_kept:
        t_round = round(t, 2)
        if t_round not in seen_times:
            seen_times.add(t_round)
            all_frames.append((t, frame))
    
    # Add fixed frames for coverage
    for t, frame in fixed_kept:
        t_round = round(t, 2)
        if t_round not in seen_times:
            seen_times.add(t_round)
            all_frames.append((t, frame))
    
    # Sort by time
    all_frames.sort(key=lambda x: x[0])
    
    # Downsample if too many
    if len(all_frames) > max_frames:
        pick = np.linspace(0, len(all_frames) - 1, max_frames).round().astype(int)
        all_frames = [all_frames[i] for i in sorted(set(pick))]

    frames = []
    for i, (t, frame) in enumerate(all_frames):
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(dict(frame_id=i, t=round(t, 2), path=str(p)))
    return frames


def pick_for_vision(frames: list[dict], out_dir: Path, n: int = 8, side: int = 1024) -> list[dict]:
    if not frames:
        return []
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, min(n, len(frames)))})
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = []
    for i in idx:
        im = cv2.imread(frames[i]["path"])
        s = side / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = out_dir / f"v_{frames[i]['frame_id']:03d}.jpg"
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        picked.append(dict(t=frames[i]["t"], path=str(p)))
    return picked


def sample_frames_parallel(video: Path, out_dir: Path, fixed_hz: float = 1.0,
                           max_frames: int = 30, max_side: int = 1280) -> list[dict]:
    return sample_frames(video, out_dir, max_frames=max_frames, fixed_hz=fixed_hz, max_side=max_side)