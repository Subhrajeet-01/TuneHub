from langchain_core.tools import tool
from pydantic import BaseModel, Field

class PlaylistBuilderInput(BaseModel):
    tracks: list[dict] = Field(description="Tracks with mood_score field, sorted by mood relevance")
    venue_type: str = Field(description="Type of venue e.g. 'gym', 'cafe', 'party'")
    energy_preference: str = Field(description="Energy preference e.g. 'low', 'medium', 'high'")
    limit: int = Field(default=8, description="Number of tracks to include in the playlist")

#BPM ranges that works per Venue Type
_VENUE_BPM_RANGES = {
    "cafe":      (85, 120),
    "gym":       (125, 160),
    "spa":       (60,  90),
    "restaurant":(90, 115),
    "retail":    (100, 130),
    "hotel":     (70, 110),
}

@tool(args_schema=PlaylistBuilderInput)
def playlist_builder_tool(
    tracks: list[dict],
    venue_type: str,
    energy_preference: str, 
    limit: int = 8
) -> list[dict]:
    """Rank and assemble a final playlist using weighted Scoring"""

    bpm_min, bpm_max = _VENUE_BPM_RANGES.get(venue_type, (80, 140))
    energy_weights = {"low": 0.2, "medium": 0.5, "high": 0.8}
    target_energy_score = energy_weights.get(energy_preference, 0.5)

    scored = []
    for track in tracks:
        mood_score = track.get("mood_score", 0.5)
        bpm = track.get("bpm", 100)
        if bpm_min <= bpm <=bpm_max:
            bpm_score = 1.0
        else:
            distance = min(abs(bpm - bpm_min), abs(bpm - bpm_max))
            bpm_score = max(0.0, 1.0 - (distance / 40))  # Penalize tracks far from target BPM range
        
        track_energy = track.get("energy", "medium")
        energy_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        track_energy_score = energy_map.get(track_energy, 0.5)
        energy_score = 1.0 - abs(target_energy_score - track_energy_score)

        # Weighted Final Score
        final_score = (
            (mood_score * 0.5) + 
            (bpm_score * 0.3) + 
            (energy_score * 0.2)
        )

        scored.append({**track, "final_score": round(final_score, 4)})
    
    ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    return ranked[:limit] 