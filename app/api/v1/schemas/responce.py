from pydantic import BaseModel, Field
from typing import Optional

class TrackSchema(BaseModel):
    id: str
    title: str
    artist: str
    bpm: int
    energy: str
    genre: str
    mood_score: Optional[float] = None
    final_score: Optional[float] = None

class RecommendResponse(BaseModel):
    playlist_id: str = Field(description="Unique identifier for the recommended playlist")
    tracks: list[TrackSchema] = Field(description="List of recommended tracks with details and scores")
    agent_reasoning: str
    session_id: str
    total_tracks: int
    venue_type: str

class FeedbackResponse(BaseModel):
    status: str
    message: str

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    dependencies: dict[str, str]