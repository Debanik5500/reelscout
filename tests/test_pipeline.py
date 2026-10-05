"""Tests for ReelScout pipeline."""

import pytest
from PIL import Image
import io

from reelscout.pipeline import analyze_multi_post, analyze_reel_with_storage
from reelscout.storage import get_phone_storage


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    images = []
    for color in ['red', 'green', 'blue']:
        img = Image.new('RGB', (200, 200), color=color)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        images.append(buf.getvalue())
    return images


class TestAnalyzeMultiPost:
    """Test analyze_multi_post function."""

    def test_analyze_multi_post_basic(self, sample_images):
        """Test basic multi-post analysis."""
        result = analyze_multi_post(sample_images[:2], job_id="test_multi_basic")
        
        assert result.job_id == "test_multi_basic"
        assert result.status == "completed"
        assert isinstance(result.summary, str)
        assert len(result.source_frames) == 2
        assert isinstance(result.timings, dict)
        assert "total" in result.timings

    def test_analyze_multi_post_single(self, sample_images):
        """Test single image analysis."""
        result = analyze_multi_post(sample_images[:1], job_id="test_multi_single")
        
        assert result.job_id == "test_multi_single"
        assert result.status == "completed"
        assert len(result.source_frames) == 1

    def test_analyze_multi_post_frames_saved(self, sample_images):
        """Test that frames are saved to phone storage."""
        storage = get_phone_storage()
        storage.cleanup_job("test_multi_storage")
        
        result = analyze_multi_post(sample_images[:1], job_id="test_multi_storage")
        
        # Check frames exist in storage
        frames = storage.get_frames("test_multi_storage")
        assert len(frames) == 1
        assert frames[0].local_path is not None
        
        # Check result saved
        loaded = storage.load_result("test_multi_storage")
        assert loaded is not None
        assert loaded.job_id == "test_multi_storage"
        
        storage.cleanup_job("test_multi_storage")

    def test_analyze_multi_post_payload_format(self, sample_images):
        """Test that payload format matches iOS ReelResult."""
        result = analyze_multi_post(sample_images[:1], job_id="test_format")
        payload = result.to_dict()
        
        # Check required fields
        assert "jobID" in payload
        assert "status" in payload
        assert "summary" in payload
        assert "candidates" in payload
        assert "timings" in payload
        assert "sourceFrames" in payload
        
        # Check frame format
        frame = payload["sourceFrames"][0]
        assert "index" in frame
        assert "timestamp" in frame
        assert "localPath" in frame
        assert "ocrText" in frame
        assert "embedding" in frame


class TestAnalyzeReelWithStorage:
    """Test analyze_reel_with_storage function."""

    def test_analyze_reel_with_storage(self):
        """Test reel analysis with frame storage."""
        import cv2
        import numpy as np
        import tempfile
        from pathlib import Path
        
        # Create test video
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        video_path = Path(temp_video.name)
        temp_video.close()

        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
        for i in range(15):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()

        try:
            result = analyze_reel_with_storage(
                video_path=video_path,
                job_id="test_reel_storage",
                max_frames=5,
            )
            
            assert result.job_id == "test_reel_storage"
            assert result.status == "completed"
            assert len(result.source_frames) > 0
            assert len(result.source_frames) <= 5
            assert "frame_extraction" in result.timings
            
            # Verify frames saved to storage
            storage = get_phone_storage()
            frames = storage.get_frames("test_reel_storage")
            assert len(frames) == len(result.source_frames)
            
            storage.cleanup_job("test_reel_storage")
        finally:
            video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])