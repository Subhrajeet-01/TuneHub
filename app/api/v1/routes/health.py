from fastapi import APIRouter
from app.api.v1.schemas.responce import HealthCheckResponse
import redis as redis_lib
import chromadb
from app.config import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify API and Redis connectivity.
    Used by Cloud Run to test service is alive and ready.
    Returns Status for all dependencies.
    """
    deps = {}

    #check Redis connectivity
    try:
        redis_client = redis_lib.from_url(settings.redis_url)
        redis_client.ping()
        deps["redis"] = "Connected"
    except Exception as e:
        deps["redis"] = "Unreachable"
        print(f"Redis connection error: {e}")
    

    #Check ChromaDB connectivity
    try:
        chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        chroma_client.heartbeat()  
        deps["chroma_db"] = "Connected"
    except Exception as e:
        deps["chroma_db"] = "Unreachable"
        print(f"ChromaDB connection error: {e}")

    overall = "Healthy" if all(status == "Connected" for status in deps.values()) else "Unhealthy"

    return HealthCheckResponse(
        status=overall,
        version="1.0.0",
        dependencies=deps
    )