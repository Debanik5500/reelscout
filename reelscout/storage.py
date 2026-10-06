"""Local phone storage for frames and images - minimal space, unified structure."""

from __future__ import annotations

import hashlib
import json
import threading
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Lazy-loaded embedding model
_embedding_model = None


def get_embedding_model():
    """Lazy load CLIP model for embeddings."""
    global _embedding_model
    if _embedding_model is None:
        try:
            import torch
            import clip
            _embedding_model = clip.load("ViT-B/32", device="cpu" if not torch.cuda.is_available() else "cuda")[0]
        except Exception:
            _embedding_model = False  # Mark as unavailable
    return _embedding_model if _embedding_model is not False else None


def generate_embedding(image_path: str) -> list[float] | None:
    """Generate CLIP embedding for an image."""
    model = get_embedding_model()
    if model is None:
        return None
    try:
        import torch
        from PIL import Image
        import clip
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = clip.tokenize([""])[0]  # dummy for preprocess
        # Actually use clip's preprocess
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        image_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().tolist()
    except Exception:
        return None


@dataclass
class FrameData:
    """Matches iOS FrameData model exactly."""
    index: int
    timestamp: float
    local_path: str
    ocr_text: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["localPath"] = d.pop("local_path")
        d["ocrText"] = d.pop("ocr_text")
        return d


@dataclass
class Candidate:
    """Matches iOS Candidate model."""
    name: str
    kind: str
    confidence: float
    url: str | None = None
    evidence: list[str] | None = None
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ReelResult:
    """Matches iOS ReelResult model exactly."""
    job_id: str
    status: str
    summary: str
    candidates: list[Candidate]
    timings: dict[str, float]
    source_frames: list[FrameData]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["jobID"] = d.pop("job_id")
        d["sourceFrames"] = [f.to_dict() for f in self.source_frames]
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


class PhoneStorage:
    """Manages local phone storage for ReelScout - frames, images, and results."""

    def __init__(self, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / "ReelScoutData"
        self.base_dir = base_dir
        self.frames_dir = base_dir / "frames"
        self.images_dir = base_dir / "images"
        self.results_dir = base_dir / "results"
        self.metadata_file = base_dir / "metadata.json"
        self._meta_lock = threading.RLock()

        for d in (self.frames_dir, self.images_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Atomic, lock-guarded write: reels are processed on several threads at once."""
        with self._meta_lock:
            for _ in range(3):
                try:
                    payload = json.dumps(self.metadata)  # another thread may be adding a job mid-dump: retry
                except RuntimeError:
                    continue
                tmp = self.metadata_file.with_name(self.metadata_file.name + ".tmp")
                try:
                    tmp.write_text(payload)
                    tmp.replace(self.metadata_file)
                except Exception:
                    pass
                return

    def _hash_content(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def save_frame(self, job_id: str, frame_idx: int, timestamp: float, image_bytes: bytes, ocr_text: str | None = None, embedding: list[float] | None = None) -> FrameData:
        """Save a frame from reel extraction to local storage."""
        content_hash = self._hash_content(image_bytes)
        ext = ".jpg"
        filename = f"{job_id}_frame_{frame_idx:04d}_{content_hash}{ext}"
        local_path = self.frames_dir / filename

        if not local_path.exists():
            with open(local_path, "wb") as f:
                f.write(image_bytes)

        # Generate embedding if not provided
        if embedding is None:
            embedding = generate_embedding(str(local_path))

        frame = FrameData(
            index=frame_idx,
            timestamp=timestamp,
            local_path=str(local_path),
            ocr_text=ocr_text,
            embedding=embedding,
        )

        if job_id not in self.metadata:
            self.metadata[job_id] = {"frames": [], "images": [], "created": datetime.now().isoformat()}
        self.metadata[job_id]["frames"].append(frame.to_dict())
        self._save_metadata()

        return frame

    def save_image(self, job_id: str, image_idx: int, image_bytes: bytes, ocr_text: str | None = None, embedding: list[float] | None = None) -> FrameData:
        """Save a post image from multi-post carousel to local storage."""
        content_hash = self._hash_content(image_bytes)
        ext = ".jpg"
        filename = f"{job_id}_img_{image_idx:04d}_{content_hash}{ext}"
        local_path = self.images_dir / filename

        if not local_path.exists():
            with open(local_path, "wb") as f:
                f.write(image_bytes)

        # Generate embedding if not provided
        if embedding is None:
            embedding = generate_embedding(str(local_path))

        frame = FrameData(
            index=image_idx,
            timestamp=0.0,
            local_path=str(local_path),
            ocr_text=ocr_text,
            embedding=embedding,
        )

        if job_id not in self.metadata:
            self.metadata[job_id] = {"frames": [], "images": [], "created": datetime.now().isoformat()}
        self.metadata[job_id]["images"].append(frame.to_dict())
        self._save_metadata()

        return frame

    def get_frames(self, job_id: str) -> list[FrameData]:
        """Retrieve all frames/images for a job from local storage."""
        if job_id not in self.metadata:
            return []

        frames = []
        for f_data in self.metadata[job_id].get("frames", []):
            frames.append(FrameData(
                index=f_data["index"],
                timestamp=f_data["timestamp"],
                local_path=f_data["localPath"],
                ocr_text=f_data.get("ocrText"),
                embedding=f_data.get("embedding"),
            ))
        for f_data in self.metadata[job_id].get("images", []):
            frames.append(FrameData(
                index=f_data["index"],
                timestamp=f_data["timestamp"],
                local_path=f_data["localPath"],
                ocr_text=f_data.get("ocrText"),
                embedding=f_data.get("embedding"),
            ))

        frames.sort(key=lambda f: f.index)
        return frames

    def save_result(self, result: ReelResult):
        """Save analysis result to local storage."""
        result_file = self.results_dir / f"{result.job_id}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f)

        if result.job_id in self.metadata:
            self.metadata[result.job_id]["result_saved"] = True
            self.metadata[result.job_id]["result_path"] = str(result_file)
        self._save_metadata()

    def load_result(self, job_id: str) -> ReelResult | None:
        """Load analysis result from local storage."""
        if job_id not in self.metadata:
            return None

        result_path = self.metadata[job_id].get("result_path")
        if not result_path or not Path(result_path).exists():
            return None

        try:
            with open(result_path) as f:
                data = json.load(f)
            return self._dict_to_result(data)
        except Exception:
            return None

    def _dict_to_result(self, data: dict[str, Any]) -> ReelResult:
        frames = []
        for f in data.get("sourceFrames", []):
            frames.append(FrameData(
                index=f["index"],
                timestamp=f["timestamp"],
                local_path=f["localPath"],
                ocr_text=f.get("ocrText"),
                embedding=f.get("embedding"),
            ))

        candidates = []
        for c in data.get("candidates", []):
            candidates.append(Candidate(
                name=c["name"],
                kind=c["kind"],
                confidence=c["confidence"],
                url=c.get("url"),
                evidence=c.get("evidence"),
                verification=c.get("verification"),
            ))

        return ReelResult(
            job_id=data["jobID"],
            status=data["status"],
            summary=data["summary"],
            candidates=candidates,
            timings=data.get("timings", {}),
            source_frames=frames,
        )

    def cleanup_job(self, job_id: str):
        """Remove all local files for a job (optional cleanup)."""
        if job_id not in self.metadata:
            return

        for f in self.metadata[job_id].get("frames", []):
            try:
                Path(f["localPath"]).unlink(missing_ok=True)
            except Exception:
                pass
        for f in self.metadata[job_id].get("images", []):
            try:
                Path(f["localPath"]).unlink(missing_ok=True)
            except Exception:
                pass

        result_path = self.metadata[job_id].get("result_path")
        if result_path:
            try:
                Path(result_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Sidecars written next to the result (context, key-moment cache, ...).
        for sidecar in self.results_dir.glob(f"{job_id}.*"):
            try:
                sidecar.unlink(missing_ok=True)
            except Exception:
                pass

        del self.metadata[job_id]
        self._save_metadata()

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage usage statistics."""
        total_frames = 0
        total_images = 0
        total_size = 0

        for job_data in self.metadata.values():
            for f in job_data.get("frames", []):
                total_frames += 1
                p = Path(f["localPath"])
                if p.exists():
                    total_size += p.stat().st_size
            for f in job_data.get("images", []):
                total_images += 1
                p = Path(f["localPath"])
                if p.exists():
                    total_size += p.stat().st_size

        return {
            "total_frames": total_frames,
            "total_images": total_images,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


_global_storage: PhoneStorage | None = None


def get_phone_storage() -> PhoneStorage:
    global _global_storage
    if _global_storage is None:
        _global_storage = PhoneStorage()
    return _global_storage