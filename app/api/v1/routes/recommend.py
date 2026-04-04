from fastapi import APIRouter, HTTPException, Request
from app.api.v1.schemas.request import RecommendRequest
from app.api.v1.schemas.responce import RecommendResponse, TrackSchema
from app.agents.state import AgentState
from langchain_core.messages import HumanMessage
import asyncio
import uuid

router = APIRouter(tags=["Recommendations"])

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest, req: Request):
    """
    Main recommendation endpoints.
    Take Natural language query and venue context, 
    return an AI-curated playlist.
    """

    agent = req.app.state.agent  # Get the pre-initialized agent from app state

    # Build initial agent state
    initial_state: AgentState = {
        "messages": [HumanMessage(request.query)],
        "session_id": request.session_id,
        "venue_context": {
            "venue_type": request.venue_context.venue_type,
            "time_of_day": request.venue_context.time_of_day,
            "energy_preference": request.venue_context.energy_preference,
            "session_duration": request.venue_context.session_duration,
        },
        "search_results": [],
        "mood_scores": {},
        "final_playlist": [],
        "plan": "",
        "reasoning_trace": [],
        "memory_context": {},
        "next_step": "",
        "error": None,
        "session_updated": False,
    }

    # Run the Agent
    try:
        result = await asyncio.to_thread(agent.invoke, initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=("Agent Failed",str(e)))
    
    if not result["final_playlist"]:
        raise HTTPException(
            status_code=422, 
            detail="Agent could not generate a playlist for this request."
        )
        
    # Build response
    tracks = [TrackSchema(**track) for track in result["final_playlist"]]
    playlist_id = f"pl_{uuid.uuid4().hex[:8]}"

    # Extract agent reasoning from last assistant message
    agent_reasoning = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            agent_reasoning += msg.content
            break
    
    return RecommendResponse(
        playlist_id=playlist_id,
        tracks=tracks,
        agent_reasoning=agent_reasoning,
        session_id=request.session_id,
        total_tracks=len(tracks),
        venue_type=request.venue_context.venue_type,
    )