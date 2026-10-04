"""OCR with automatic engine choice (RapidOCR if installed, else Tesseract).

Two passes per frame: the whole frame, plus the top and bottom strips upscaled 3x,
because browser address bars are where the URL lives and they are tiny on a phone video.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import cv2

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            r = RapidOCR()
            _engine = ("rapid", r)
        except Exception:
            import pytesseract
            _engine = ("tess", pytesseract)
    return _engine


def _read(img) -> list[tuple[str, float]]:
    kind, eng = _get_engine()
    if kind == "rapid":
        res, _ = eng(img)
        return [(t, float(c)) for _, t, c in (res or [])]
    data = eng.image_to_data(img, config="--psm 11", output_type=eng.Output.DICT)
    lines: dict = {}
    for i, w in enumerate(data["text"]):
        w = w.strip()
        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue
        if w and conf >= 40:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((w, conf))
    return [(" ".join(w for w, _ in ws), sum(c for _, c in ws) / len(ws) / 100) for ws in lines.values()]


def ocr_frame(path: str) -> list[tuple[str, float]]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    up = lambda im, f: cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    out = _read(up(img, 2) if w < 1100 else img)
    out += _read(up(img[: int(h * 0.16)], 3))   # top strip (address bar / tab titles)
    out += _read(up(img[int(h * 0.84):], 3))    # bottom strip (mobile browser bars / captions)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _ocr_single(f: dict) -> tuple[float, list[tuple[str, float]]]:
    """Process one frame, return (timestamp, results)."""
    return f["t"], ocr_frame(f["path"])


def ocr_frames(frames: list[dict], workers: int = 1) -> list[dict]:
    """Returns de-duplicated lines: {text, t (first seen), n_frames, conf}.

    If workers > 1, processes frames in parallel using ThreadPoolExecutor.
    """
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_ocr_single, frames))
    else:
        results = [_ocr_single(f) for f in frames]

    seen: dict[str, dict] = {}
    for t, texts in results:
        in_this_frame = set()
        for text, conf in texts:
            text = text.strip()
            k = _norm(text)
            if len(k) < 3 or k in in_this_frame:
                continue
            in_this_frame.add(k)
            if k in seen:
                seen[k]["n_frames"] += 1
                seen[k]["conf"] = max(seen[k]["conf"], conf)
            else:
                seen[k] = dict(text=text, t=t, n_frames=1, conf=round(conf, 2))
    return list(seen.values())
