from fastembed import TextEmbedding
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field

print("Loading Mood Analyser Model...")
_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
print("Model Loaded.")

_MOOD_ANCHORS = {
    "energetic": "high energy powerful driving intense pumping",
    "calm": "peaceful relaxing gentle soft soothing tranquil",
    "upbeat": "happy cheerful positive bright optimistic fun",
    "focused": "concentrated deep steady rhythmic productive",
    "romantic": "warm intimate emotional tender loving",
    "melancholic": "sad reflective nostalgic longing bittersweet",
}

def _embed(text: str):
    return list(_MODEL.embed([text]))[0]

_ANCHOR_EMBEDDINGS = {
    mood: _embed(desc)
    for mood, desc in _MOOD_ANCHORS.items()
}

# cosine similarity - just compute manually:
def _cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# TextEmbedding returns generator, so we need a helper to get single embedding out easily
def embed_one(model, text):
    return next(model.embed([text]))

class MoodAnalyzerInput(BaseModel):
    tracks: list[dict] = Field(description="List of tracks dicts with mood_tags field") 
    target_mood: str = Field(description="The mood e.g. 'energetic', 'calm', etc.")

@tool(args_schema=MoodAnalyzerInput)
def mood_analyzer_tool(tracks: list[dict], target_mood: str) -> list[dict]:
    """Score tracks based on how well their mood_tags match the target mood."""
    
    if target_mood not in _ANCHOR_EMBEDDINGS:
        target_embedding = embed_one(_MODEL, target_mood)
    else:
        target_embedding = _ANCHOR_EMBEDDINGS[target_mood]

    scored_tracks = []
    for track in tracks:
        mood_text = " ".join(track.get("mood_tags", []))
        if not mood_text:
            mood_text = f"{track.get('genre', '')} {track.get('energy', '')}"
        
        track_embedding = embed_one(_MODEL, mood_text)

        score = _cosine_sim(target_embedding, track_embedding)

        scored_tracks.append({
            **track,
            "mood_score": float(round(score, 4))
        })

    return sorted(scored_tracks, key=lambda x: x["mood_score"], reverse=True)

