from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage  
import operator

class VenueContext(TypedDict):
    venue_type: str                    # cafe, gym, hotel, retail, restaurant, spa
    time_of_day: str                   # morning, afternoon, evening, night              
    energy_preference: str             # low, medium, high
    session_duration: str              # minutes

class Trackresult(TypedDict):
    id: str
    title: str
    artist: str
    bpm: int
    mood_score: float
    energy: str
    genre: str

class AgentState(TypedDict):
    #core conversation
    messages: Annotated[list[BaseMessage], operator.add]

    #session context
    session_id: str
    venue_context: VenueContext

    #Tools Output
    search_results: list[Trackresult]
    mood_score: dict[str, float]
    final_playlist: list[Trackresult]

    #Agent Reasoning
    plan: str
    reasoning_trace: list[str]

    #control_flow
    next_step: str
    error: Optional[str]

    #memory
    memory_context: dict
    session_updated: bool

