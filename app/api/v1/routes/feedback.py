from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.request import FeedbackRequest
from app.api.v1.schemas.responce import FeedbackResponse
from app.memory.session import SessionMemory

router = APIRouter(tags=["Feedback"])

_FEEDBACK_SIGNALS = {
    "too slow": ("energy_preference", "high"),
    "too fast": ("energy_preference", "low"),
    "too loud":    ("energy_preference", "low"),
    "too calm":    ("energy_preference", "high"),
    "too quiet":   ("energy_preference", "medium"),
    "too intense": ("energy_preference", "low"),
}

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit Feedback for a recommended playlist.
    Updates the user's session context based on feedback signals to improve future recommendations.
    """

    try:
        #parse feedback signal from text.
        if request.feedback_text:
            feedback_lower = request.feedback_text.lower()
            for signal, (pref_key, pref_value) in _FEEDBACK_SIGNALS.items():
                if signal in feedback_lower:
                    # This is where update_preference is called
                    SessionMemory.update_preference(
                        session_id=request.session_id,
                        key=pref_key,
                        value=pref_value
                    )
                    break
            
        # Low rating (1-2) — store as negative signal
        if request.rating <= 2:
            SessionMemory.update_preference(
                session_id=request.session_id,
                key="last_rating",
                value="negative"
            )
        elif request.rating >= 4:
            SessionMemory.update_preference(
                session_id=request.session_id,
                key="last_rating",
                value="positive"
            )

        return FeedbackResponse(
            status="memory_updated",
            message=f"Feedback recorded. Future recommendations will adjust accordingly."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))