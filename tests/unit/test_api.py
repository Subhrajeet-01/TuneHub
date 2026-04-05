import pytest
from unittest.mock import patch, MagicMock

class TestRecommendEndpoint:

    def test_health_returns_200(self, client):
        """Health endpoint must always return 200."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "dependencies" in data

    def test_recommend_invalid_venue_returns_422(self, client, recommend_payload):
        """Invalid venue type must return 422 automatically."""
        recommend_payload["venue_context"]["venue_type"] = "nightclub"
        response = client.post("/v1/recommend", json=recommend_payload)
        assert response.status_code == 422

    def test_recommend_missing_query_returns_422(self, client, recommend_payload):
        """Missing query field must return 422."""
        del recommend_payload["query"]
        response = client.post("/v1/recommend", json=recommend_payload)
        assert response.status_code == 422

    def test_recommend_tracks_have_required_fields(self, client, recommend_payload):
        """Every track in response must have id, title, artist, bpm."""
        response = client.post("/v1/recommend", json=recommend_payload)
        assert response.status_code == 200
        for track in response.json()["tracks"]:
            assert "id" in track
            assert "title" in track
            assert "artist" in track
            assert "bpm" in track
        
    def test_feedback_valid_submission_returns_200(self, client):
        """Valid feedback must return 200 with memory_updated status."""
        response = client.post("/v1/feedback", json={
            "playlist_id": "pl_test_001",
            "session_id": "test_session_001",
            "rating": 4,
            "feedback_text": "great selection"
        })
        assert response.status_code == 200
        assert response.json()["status"] == "memory_updated"

    def test_feedback_invalid_rating_returns_422(self, client):
        """Rating outside 1-5 must be rejected."""
        response = client.post("/v1/feedback", json={
            "playlist_id": "pl_001",
            "session_id": "test_001",
            "rating": 10
        })
        assert response.status_code == 422