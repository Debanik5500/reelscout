"""Tests for ReelScout API endpoints."""

import pytest
import base64
import io
from PIL import Image
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_b64():
    """Create a sample base64 encoded image."""
    img = Image.new('RGB', (200, 200), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes."""
    img = Image.new('RGB', (200, 200), color='green')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "workers" in data
        assert "queue_len" in data


class TestMultiPostEndpoint:
    """Test multi-post analysis endpoint."""

    def test_analyze_multi_post_single(self, client, sample_image_b64):
        """Test single post analysis (index=0)."""
        response = client.post(
            "/api/v1/analyze-multi-post",
            json={
                "images": [sample_image_b64],
                "index": 0,
                "job_id": "test_single_123",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_single_123"
        assert data["status"] == "queued"

    def test_analyze_multi_post_carousel(self, client, sample_image_b64):
        """Test carousel analysis (index>=1)."""
        response = client.post(
            "/api/v1/analyze-multi-post",
            json={
                "images": [sample_image_b64, sample_image_b64, sample_image_b64],
                "index": 1,
                "job_id": "test_carousel_123",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_carousel_123"
        assert data["status"] == "queued"

    def test_analyze_multi_post_invalid_image(self, client):
        """Test with invalid base64."""
        response = client.post(
            "/api/v1/analyze-multi-post",
            json={
                "images": ["not_valid_base64"],
                "index": 0,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Invalid image data" in data["error"]

    def test_analyze_multi_post_empty(self, client):
        """Test with no images."""
        response = client.post(
            "/api/v1/analyze-multi-post",
            json={
                "images": [],
                "index": 0,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "No valid images" in data["error"]


class TestReelStorageEndpoint:
    """Test reel with storage endpoint."""

    def test_analyze_reel_storage(self, client, sample_image_bytes):
        """Test reel analysis with frame storage."""
        # Create a dummy video file
        import cv2
        import numpy as np
        import tempfile
        
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        video_path = temp_video.name
        temp_video.close()

        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        for i in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()

        try:
            with open(video_path, "rb") as f:
                response = client.post(
                    "/api/v1/analyze-reel-storage",
                    files={"video": ("test.mp4", f, "video/mp4")},
                    data={"job_id": "test_reel_123", "max_frames": 5},
                )
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test_reel_123"
            assert data["status"] in ("completed", "error")
        finally:
            import os
            os.unlink(video_path)


class TestInstagramEndpoint:
    """Test Instagram URL analysis endpoint."""

    def test_analyze_instagram_url(self, client):
        """Test Instagram URL endpoint queues job."""
        response = client.post(
            "/api/v1/analyze-instagram",
            json={
                "url": "https://www.instagram.com/reel/ABC123/",
                "job_id": "test_ig_123",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_ig_123"
        assert data["status"] == "queued"


class TestStorageEndpoints:
    """Test storage management endpoints."""

    def test_get_storage_stats(self, client):
        """Test storage stats endpoint."""
        response = client.get("/api/v1/storage/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_frames" in data
        assert "total_images" in data
        assert "total_size_bytes" in data
        assert "total_size_mb" in data

    def test_cleanup_storage(self, client):
        """Test storage cleanup endpoint."""
        response = client.delete("/api/v1/storage/test_cleanup_123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleaned"
        assert data["job_id"] == "test_cleanup_123"


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        # Check for some expected metrics
        content = response.text
        assert "reelscout_requests_total" in content
        assert "reelscout_request_duration_seconds" in content


class TestAuth:
    """Test API key authentication."""

    def test_without_api_key(self, client, sample_image_b64):
        """Test without API key when not configured (should work)."""
        # By default API_KEY is empty, so auth is disabled
        response = client.post(
            "/api/v1/analyze-multi-post",
            json={"images": [sample_image_b64], "index": 0}
        )
        assert response.status_code == 200

    def test_with_invalid_api_key(self, client, sample_image_b64):
        """Test with invalid API key when configured."""
        # We can't easily test this without setting API_KEY env var
        # This is a placeholder for when API_KEY is set
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])