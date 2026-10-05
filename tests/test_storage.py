"""Tests for ReelScout storage module."""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import io

from reelscout.storage import (
    PhoneStorage,
    FrameData,
    Candidate,
    ReelResult,
    get_phone_storage,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PhoneStorage(Path(tmpdir) / "ReelScoutData")
        yield storage


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes."""
    img = Image.new('RGB', (200, 200), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


class TestPhoneStorage:
    """Test PhoneStorage class."""

    def test_init_creates_directories(self, temp_storage):
        """Test that initialization creates all required directories."""
        assert temp_storage.base_dir.exists()
        assert temp_storage.frames_dir.exists()
        assert temp_storage.images_dir.exists()
        assert temp_storage.results_dir.exists()

    def test_save_and_get_frame(self, temp_storage, sample_image_bytes):
        """Test saving and retrieving a frame."""
        frame = temp_storage.save_frame("job_123", 0, 1.5, sample_image_bytes, ocr_text="Test OCR")
        
        assert frame.index == 0
        assert frame.timestamp == 1.5
        assert frame.ocr_text == "Test OCR"
        assert frame.local_path is not None
        assert Path(frame.local_path).exists()

        frames = temp_storage.get_frames("job_123")
        assert len(frames) == 1
        assert frames[0].index == 0
        assert frames[0].ocr_text == "Test OCR"

    def test_save_and_get_image(self, temp_storage, sample_image_bytes):
        """Test saving and retrieving an image (for carousel)."""
        frame = temp_storage.save_image("job_456", 1, sample_image_bytes, ocr_text="Carousel OCR")
        
        assert frame.index == 1
        assert frame.timestamp == 0.0
        assert frame.ocr_text == "Carousel OCR"

        frames = temp_storage.get_frames("job_456")
        assert len(frames) == 1
        assert frames[0].index == 1

    def test_mixed_frames_and_images(self, temp_storage, sample_image_bytes):
        """Test storing both frames and images in same job."""
        temp_storage.save_frame("job_mixed", 0, 0.5, sample_image_bytes)
        temp_storage.save_image("job_mixed", 1, sample_image_bytes)
        temp_storage.save_frame("job_mixed", 2, 1.5, sample_image_bytes)
        
        frames = temp_storage.get_frames("job_mixed")
        assert len(frames) == 3
        # Should be sorted by index
        assert frames[0].index == 0
        assert frames[1].index == 1
        assert frames[2].index == 2

    def test_deduplication(self, temp_storage, sample_image_bytes):
        """Test that identical images have same content hash."""
        temp_storage.save_frame("job_dedup", 0, 0.0, sample_image_bytes)
        temp_storage.save_frame("job_dedup", 1, 1.0, sample_image_bytes)  # Same content
        
        stats = temp_storage.get_storage_stats()
        assert stats["total_frames"] == 2
        assert stats["total_size_bytes"] > 0
        # Both frames should have same content hash in filename
        frames = temp_storage.get_frames("job_dedup")
        # Extract hash from filename
        hash_0 = Path(frames[0].local_path).stem.split("_")[-1]
        hash_1 = Path(frames[1].local_path).stem.split("_")[-1]
        assert hash_0 == hash_1  # Same content = same hash

    def test_save_and_load_result(self, temp_storage, sample_image_bytes):
        """Test saving and loading analysis result."""
        frame = temp_storage.save_frame("job_result", 0, 0.0, sample_image_bytes)
        
        candidate = Candidate(
            name="TestProduct",
            kind="product",
            confidence=0.9,
            url="https://example.com",
            evidence=[{"source": "ocr", "text": "TestProduct"}],
        )
        
        result = ReelResult(
            job_id="job_result",
            status="completed",
            summary="Found TestProduct",
            candidates=[candidate],
            timings={"total": 1.0, "ocr": 0.5},
            source_frames=[frame],
        )
        
        temp_storage.save_result(result)
        loaded = temp_storage.load_result("job_result")
        
        assert loaded is not None
        assert loaded.job_id == "job_result"
        assert loaded.status == "completed"
        assert loaded.summary == "Found TestProduct"
        assert len(loaded.candidates) == 1
        assert loaded.candidates[0].name == "TestProduct"
        assert len(loaded.source_frames) == 1

    def test_load_nonexistent_result(self, temp_storage):
        """Test loading a result that doesn't exist."""
        loaded = temp_storage.load_result("nonexistent")
        assert loaded is None

    def test_cleanup_job(self, temp_storage, sample_image_bytes):
        """Test cleaning up a job."""
        temp_storage.save_frame("job_cleanup", 0, 0.0, sample_image_bytes)
        temp_storage.save_image("job_cleanup", 1, sample_image_bytes)
        
        candidate = Candidate(name="Test", kind="product", confidence=0.5)
        result = ReelResult(
            job_id="job_cleanup",
            status="completed",
            summary="Test",
            candidates=[candidate],
            timings={},
            source_frames=[],
        )
        temp_storage.save_result(result)
        
        # Verify files exist
        stats_before = temp_storage.get_storage_stats()
        assert stats_before["total_frames"] == 1
        assert stats_before["total_images"] == 1
        
        temp_storage.cleanup_job("job_cleanup")
        
        # Verify files removed
        stats_after = temp_storage.get_storage_stats()
        assert stats_after["total_frames"] == 0
        assert stats_after["total_images"] == 0
        assert temp_storage.load_result("job_cleanup") is None

    def test_storage_stats(self, temp_storage, sample_image_bytes):
        """Test storage statistics."""
        temp_storage.save_frame("job_stats", 0, 0.0, sample_image_bytes)
        temp_storage.save_image("job_stats", 1, sample_image_bytes)
        
        stats = temp_storage.get_storage_stats()
        assert stats["total_frames"] == 1
        assert stats["total_images"] == 1
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] >= 0


class TestFrameData:
    """Test FrameData model."""

    def test_to_dict(self):
        """Test FrameData to_dict conversion."""
        frame = FrameData(
            index=5,
            timestamp=2.5,
            local_path="/path/to/frame.jpg",
            ocr_text="Hello World",
            embedding=[0.1, 0.2, 0.3],
        )
        
        d = frame.to_dict()
        assert d["index"] == 5
        assert d["timestamp"] == 2.5
        assert d["localPath"] == "/path/to/frame.jpg"
        assert d["ocrText"] == "Hello World"
        assert d["embedding"] == [0.1, 0.2, 0.3]


class TestCandidate:
    """Test Candidate model."""

    def test_to_dict(self):
        """Test Candidate to_dict conversion."""
        candidate = Candidate(
            name="TestProduct",
            kind="product",
            confidence=0.95,
            url="https://example.com",
            evidence=[{"source": "ocr", "text": "TestProduct"}],
            verification={"ok": True, "url": "https://example.com"},
        )
        
        d = candidate.to_dict()
        assert d["name"] == "TestProduct"
        assert d["kind"] == "product"
        assert d["confidence"] == 0.95
        assert d["url"] == "https://example.com"
        assert d["evidence"] == [{"source": "ocr", "text": "TestProduct"}]
        assert d["verification"] == {"ok": True, "url": "https://example.com"}


class TestReelResult:
    """Test ReelResult model."""

    def test_to_dict(self, sample_image_bytes):
        """Test ReelResult to_dict conversion."""
        frame = FrameData(index=0, timestamp=0.0, local_path="/path.jpg")
        candidate = Candidate(name="Test", kind="product", confidence=0.8)
        
        result = ReelResult(
            job_id="test_job",
            status="completed",
            summary="Test summary",
            candidates=[candidate],
            timings={"total": 1.0},
            source_frames=[frame],
        )
        
        d = result.to_dict()
        assert d["jobID"] == "test_job"
        assert d["status"] == "completed"
        assert d["summary"] == "Test summary"
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["name"] == "Test"
        assert len(d["sourceFrames"]) == 1
        assert d["sourceFrames"][0]["localPath"] == "/path.jpg"


class TestGlobalStorage:
    """Test global storage singleton."""

    def test_get_phone_storage_singleton(self):
        """Test that get_phone_storage returns same instance."""
        storage1 = get_phone_storage()
        storage2 = get_phone_storage()
        assert storage1 is storage2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])