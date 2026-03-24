from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field

print("Loading Mood Analyser Model...")
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model Loaded.")

_MOOD_ANCHORS = {
    "energetic": "high energy powerful driving intense pumping",
    "calm": "peaceful relaxing gentle soft soothing tranquil",
    "upbeat": "happy cheerful positive bright optimistic fun",
    "focused": "concentrated deep steady rhythmic productive",
    "romantic": "warm intimate emotional tender loving",
    "melancholic": "sad reflective nostalgic longing bittersweet",
}

# Pre-compute anchor embeddings once
_ANCHOR_EMBEDDINGS = {
    mood: _MODEL.encode([desc])[0]
    for mood, desc in _MOOD_ANCHORS.items()
}

class MoodAnalyzerInput(BaseModel):
    tracks: list[dict] = Field(description="List of tracks dicts with mood_tags field") 
    target_mood: str = Field(description="The mood e.g. 'energetic', 'calm', etc.")

@tool(args_schema=MoodAnalyzerInput)
def mood_analyzer_tool(tracks: list[dict], target_mood: str) -> list[dict]:
    """Score tracks based on how well their mood_tags match the target mood."""
    
    if target_mood not in _ANCHOR_EMBEDDINGS:
        target_embedding = _MODEL.encode([target_mood])[0]
    else:
        target_embedding = _ANCHOR_EMBEDDINGS[target_mood]

    scored_tracks = []
    for track in tracks:
        mood_text = " ".join(track.get("mood_tags", []))
        if not mood_text:
            mood_text = f"{track.get('genre', '')} {track.get('energy', '')}"
        
        track_embedding = _MODEL.encode([mood_text])[0]

        score = cosine_similarity(
            target_embedding.reshape(1, -1),
            track_embedding.reshape(1, -1)
        )[0][0]

        scored_tracks.append({
            **track,
            "mood_score": float(round(score, 4))
        })

    return sorted(scored_tracks, key=lambda x: x["mood_score"], reverse=True)

