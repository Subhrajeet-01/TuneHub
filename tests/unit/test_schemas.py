import pytest
from pydantic import ValidationError
from app.api.v1.schemas.request import RecommendRequest, FeedbackRequest

class TestRecommendResponseSchema:

    def test_valid_request_passes(self, recommend_payload):
        """A well-formed request must parse without errors."""
        req = RecommendRequest(**recommend_payload)
        assert req.query == recommend_payload["query"]
        assert req.session_id == recommend_payload["session_id"]

    def test_query_too_short_rejected(self, recommend_payload):
        """Query under 3 characters must be rejected."""
        recommend_payload["query"] = "hi"
        with pytest.raises(ValidationError) as exc:
            RecommendRequest(**recommend_payload)
        assert "query" in str(exc.value)

    def test_invalid_venue_type_rejected(self, recommend_payload):
        """Venue types not in the allowed list must be rejected."""
        recommend_payload["venue_context"]["venue_type"] = "nightclub"
        with pytest.raises(ValidationError):
            RecommendRequest(**recommend_payload)

    def test_invalid_energy_level_rejected(self, recommend_payload):
        """Energy level must be low/medium/high only."""
        recommend_payload["venue_context"]["energy_preference"] = "extreme"
        with pytest.raises(ValidationError):
            RecommendRequest(**recommend_payload)

    def test_limit_above_max_rejected(self, recommend_payload):
        """Limit above 20 must be rejected."""
        recommend_payload["limit"] = 100
        with pytest.raises(ValidationError):
            RecommendRequest(**recommend_payload)

class TestFeedbackRequestSchema:

    def test_valid_feedback_passes(self):
        req = FeedbackRequest(
            playlist_id="pl_001",
            session_id="test_001",
            rating=4,
            feedback_text="Great selection!"
        )
        assert req.rating == 4

    def test_rating_below_1_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                playlist_id="pl_001",
                session_id="test_001",
                rating=0  # below minimum
            )

    def test_rating_above_5_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                playlist_id="pl_001",
                session_id="test_001",
                rating=6  # above maximum
            )