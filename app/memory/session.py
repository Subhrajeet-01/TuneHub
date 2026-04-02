import json
import redis
from app.config import get_settings

settings = get_settings()

_client = redis.from_url(settings.redis_url, decode_responses=True)

SESSION_TTL_SECONDS = 86400  # 24 hour

class SessionMemory:

    @staticmethod
    def get(session_id: str) -> dict:
        """Retrieve session context from Redis by session_id."""
        raw = _client.get(f"session:{session_id}")
        if not raw:
            return {
                "history": [],
                "preferences": {},
                "last_playlist_id": []
            }
        return json.loads(raw)

    @staticmethod
    def update(session_id:str, interaction: dict) -> None:
        """
        Add a new interaction to session History.
        Keeps only last 10 interactions to manage bloating context
        """
        current = SessionMemory.get(session_id)
        current["history"].append(interaction)
        # Keep only last 10 interactions
        if len(current["history"]) > 10:
            current["history"] = current["history"][-10:]

        if "preferences" in interaction:
            current["preferences"].update(interaction["preferences"])

        if "playlist_id" in interaction:
            current["last_playlist_id"] = [interaction["playlist_id"]]

        #save back with TTL reset
        _client.setex(
            f"session:{session_id}",
            SESSION_TTL_SECONDS,
            json.dumps(current)
        )

    @staticmethod
    def update_preference(session_id: str, key: str, value) -> None:
        """Update a single preference key eg. energy_preference after feedback."""
        current = SessionMemory.get(session_id)
        current["preferences"][key] = value
        _client.setex(
            f"session:{session_id}",
            SESSION_TTL_SECONDS,
            json.dumps(current)
        )
    
    @staticmethod
    def count() -> int:
        """Utility to count how many sessions are currently stored."""
        keys = _client.keys("session:*")
        return len(keys)
    
    @staticmethod
    def clear(session_id: str) -> None:
        """Clear session context."""
        _client.delete(f"session:{session_id}")

