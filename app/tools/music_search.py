import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

#---- Input Schema ----#
class MusicSearchInput(BaseModel):
    genre: str = Field(description="Music genre e.g. electronic, jazz, ambient")
    energy_level: Literal['low', 'medium', 'high'] = Field(description="Energy level of the music")
    bpm_min: int = Field(default=60, description="Minimum BPM")
    bpm_max: int = Field(default=180, description="Maximum BPM")
    mood_tags: list[str] = Field(
        default=[],
        description="List of mood tags e.g. calm, happy, sad, energetic"
    )
    limits: int = Field(default=5, description="Max tracks to return")

#---- Load Mock data----#
DATA_PATH = Path(__file__).parent.parent / "data" / "mock_data.json"

def _load_tracks() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)
    
_ALL_TRACKS = _load_tracks()

#---- Tool Function ----#
@tool(args_schema=MusicSearchInput)
def music_search_tool(
    genre: str,
    energy_level: str,
    bpm_min: int = 60,
    bpm_max: int = 180,
    mood_tags: list[str] = [],
    limits: int = 20
) -> list[dict]:
    """search for music tracks based on genre, energy level, bpm range, and mood tags"""
    results = []

    for track in _ALL_TRACKS:
        if track["energy"] != energy_level:
            continue
        if not (bpm_min <= track["bpm"] <= bpm_max):
            continue
        if genre.lower() in track["genre"].lower():
            results.append(track)
    
    # If Filter is Too Strict. Relax mood tags

    if len(results) < 5:
        results = [
            t for t in _ALL_TRACKS
            if t["energy"] == energy_level 
            and bpm_min <= t["bpm"] <= bpm_max
        ]

    if len(results) == 0:
        results = [t for t in _ALL_TRACKS if t["energy"] == energy_level]

    return results[:limits]
