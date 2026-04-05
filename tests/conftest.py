import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

@pytest.fixture
def client():
    """
    FastAPI TestClient fixture for making API requests in tests.
    Automatically handles startup and shutdown events.
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_tracks():
    return [
        {"id": "t_001", "title": "Morning Ritual", "artist": "Bonobo",
         "genre": "electronic", "bpm": 112, "energy": "medium",
         "mood_tags": ["calm", "focused"], "valence": 0.72,
         "danceability": 0.65, "decade": "2020s", "duration_seconds": 245},
        {"id": "t_002", "title": "Power Hour", "artist": "Disclosure",
         "genre": "house", "bpm": 128, "energy": "high",
         "mood_tags": ["energetic", "intense"], "valence": 0.89,
         "danceability": 0.91, "decade": "2020s", "duration_seconds": 312},
        {"id": "t_003", "title": "Quiet Afternoon", "artist": "Nils Frahm",
         "genre": "ambient", "bpm": 72, "energy": "low",
         "mood_tags": ["calm", "peaceful"], "valence": 0.45,
         "danceability": 0.30, "decade": "2020s", "duration_seconds": 380},
        {"id": "t_004", "title": "Jazz Morning", "artist": "Melody Gardot",
         "genre": "jazz", "bpm": 88, "energy": "low",
         "mood_tags": ["calm", "romantic"], "valence": 0.55,
         "danceability": 0.35, "decade": "2020s", "duration_seconds": 245},
        {"id": "t_005", "title": "Deep Focus", "artist": "Tycho",
         "genre": "electronic", "bpm": 105, "energy": "medium",
         "mood_tags": ["focused", "calm"], "valence": 0.60,
         "danceability": 0.55, "decade": "2020s", "duration_seconds": 290},
    ]

@pytest.fixture
def sample_venue_context():
    return {
        "venue_type": "cafe",
        "time_of_day": "morning",
        "energy_preference": "medium",
        "session_duration": 120
    }

@pytest.fixture
def recommend_payload(sample_venue_context):
    return {
        "query": "upbeat morning music for a coffee shop",
        "venue_context": sample_venue_context,
        "session_id": "test_session_001",
        "limit": 5
    }