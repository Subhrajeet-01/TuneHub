import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from app.config import get_settings

settings = get_settings()

_embedding_fn = ONNXMiniLM_L6_V2()

# persistent chroma client
_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

# Single collection for all playlists.
_collection = _client.get_or_create_collection(
    name = "playlists",
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"} 
)

class SemanticMemory:

    @staticmethod
    def store_playlist(
        playlist_id: str,
        query: str,
        venue_type: str,
        track_ids: list[str],
        energy: str,
    ) -> None:
        """
        Store a successful playlist as an embedding
        The query text become the embedding — future similar queries retrieve this.
        """
        # Build rich text representation for embedding
        document = f"{query} venue:{venue_type} energy:{energy}"

        _collection.upsert(
            ids = [playlist_id],
            documents = [document],
            metadatas = [{
                "venue_type": venue_type,
                "energy": energy,
                "track_ids": ",".join(track_ids),
                "query": query
            }]
        )
    
    @staticmethod
    def retrieve_similar(
        query: str,
        venue_type: str,
        top_k: int = 5
        ) -> list[dict]:
        """
        Retrieve similar past playlists based on query and venue context.
        Used to inject as few-shot context into the planner.
        """

        if _collection.count() == 0:
            return []
        
        results = _collection.query(
            query_texts = [f"{query} venue:{venue_type}"],
            n_results=min(top_k, _collection.count()),
            where={"venue_type": venue_type}
        )

        similar = []
        for i, metadata in enumerate(results["metadatas"][0]):
            similar.append({
                "query": metadata["query"],
                "venue_type": metadata["venue_type"],
                "energy": metadata["energy"],
                "track_ids": metadata["track_ids"].split(","),
                "similarity_score": 1 - results["distances"][0][i]  # Convert cosine distance to similarity
            })
        
        return similar

    @staticmethod
    def count() -> int:
        """Utility to check how many playlists are stored."""
        return _collection.count()
    
    @staticmethod
    def clear() -> None:
        """Utility to clear all stored playlists (for testing)."""
        _client.delete_collection(name="playlists")