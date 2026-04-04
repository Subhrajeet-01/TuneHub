from pydantic import BaseModel, Field
from typing import Literal, Optional

class VenueContextSchema(BaseModel):
    venue_type: Literal["cafe", "gym", "hotel", "retail", "restaurant", "spa"]
    time_of_day: Literal["morning", "afternoon", "evening", "night"]
    energy_preference: Literal["low", "medium", "high"]
    session_duration: int = Field(
        default=60,
        ge = 15,     # Minimum 15 minutes
        le = 240,    # Maximum 4 hours
        description="Duration of the Playlist in minutes"
    )

class RecommendRequest(BaseModel):
    query: str = Field(
        min_lenght = 3,
        max_length= 500,
        description="User's natural language query describing their music needs and context",
        examples=["upbeat morning music for cafe", "energetic gym workout playlist", "calm spa relaxation music"]
    )

    venue_context: VenueContextSchema
    session_id: str = Field(
        min_length=3,
        max_length=100,
        description="Unique identifier for the user session, used for memory retrieval and context management.",
        examples=["cafe_owner_1"]
    )
    limit: int = Field(default=8, ge=1, le=20, description="Number of tracks to include in the recommended playlist")

class FeedbackRequest(BaseModel):
    playlist_id: str 
    session_id: str
    rating: int = Field(ge=1, le=5, description="User rating for the recommended playlist (1-5)")
    feedback_text: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Optional textual feedback from the user about the recommended playlist"
    )