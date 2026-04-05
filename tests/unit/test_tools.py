import pytest
from unittest.mock import patch, MagicMock
from app.tools.music_search import music_search_tool
from app.tools.mood_analyser import mood_analyzer_tool
from app.tools.playlist_builder import playlist_builder_tool

class TestMusicSearchTool:

    def test_returns_correct_energy_level(self, sample_tracks):
        """Tool must only return tracks matching requested energy level."""
        with patch("app.tools.music_search._ALL_TRACKS", sample_tracks):
            result = music_search_tool.invoke({
                "genre": "electronic",
                "energy_level": "medium",
                "bpm_min": 60,
                "bpm_max": 180,
                "limit": 10
            })
        # Every returned track must have energy == "medium"
        assert all(t["energy"] == "medium" for t in result)

    def test_respects_limit(self, sample_tracks):
        """Tool must never return more tracks than the limit."""
        with patch("app.tools.music_search._ALL_TRACKS", sample_tracks):
            result = music_search_tool.invoke({
                "genre": "electronic",
                "energy_level": "medium",
                "bpm_min": 60,
                "bpm_max": 180,
                "limit": 2
            })
        assert len(result) <= 2

    def test_bpm_range_filter(self, sample_tracks):
        """Tool must only return tracks within the BPM range."""
        with patch("app.tools.music_search._ALL_TRACKS", sample_tracks):
            result = music_search_tool.invoke({
                "genre": "electronic",
                "energy_level": "medium",
                "bpm_min": 100,
                "bpm_max": 120,
                "limit": 10
            })
        assert all(100 <= t["bpm"] <= 120 for t in result)


class TestMoodAnalyzerTool:

    def test_returns_mood_score_for_each_track(self, sample_tracks):
        """Every returned track must have a mood_score field."""
        result = mood_analyzer_tool.invoke({
            "tracks": sample_tracks,
            "target_mood": "calm"
        })
        assert all("mood_score" in t for t in result)

    def test_mood_score_range(self, sample_tracks):
        """Mood scores must be between 0 and 1."""
        result = mood_analyzer_tool.invoke({
            "tracks": sample_tracks,
            "target_mood": "energetic"
        })
        assert all(0.0 <= t["mood_score"] <= 1.0 for t in result)

    def test_sorted_by_mood_score_descending(self, sample_tracks):
        """Results must be sorted highest mood_score first."""
        result = mood_analyzer_tool.invoke({
            "tracks": sample_tracks,
            "target_mood": "calm"
        })
        scores = [t["mood_score"] for t in result]
        assert scores == sorted(scores, reverse=True)


class TestPlaylistBuilderTool:

    def test_sorted_by_final_score_descending(self, sample_tracks):
        """Results must be sorted highest final_score first."""
        tracks_with_scores = [
            {**t, "mood_score": 0.6} for t in sample_tracks
        ]
        result = playlist_builder_tool.invoke({
            "tracks": tracks_with_scores,
            "venue_type": "gym",
            "energy_preference": "high",
            "limit": 10
        })
        scores = [t["final_score"] for t in result]
        assert scores == sorted(scores, reverse=True)

    def test_handles_empty_track_list(self):
        """Builder must return empty list when given no tracks."""
        result = playlist_builder_tool.invoke({
            "tracks": [],
            "venue_type": "cafe",
            "energy_preference": "medium",
            "limit": 8
        })
        assert result == []