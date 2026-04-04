from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.agents.graph import build_graph
from app.config import get_settings
import uuid

setting = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize agent and store in app state
    print("Starting MusicMind Agent...")
    app.state.agent = build_graph()         #compile agent once at startup.
    print("Agent initialized and ready.")
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down MusicMind Agent...")

# App Definition
app = FastAPI(
    title="MusicMind Agent API",
    description="API for the MusicMind Agent - a context-aware music recommendation system.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
# Allow React Dashboard to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production, specify allowed origins for security.
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
from app.api.v1.routes import recommend, health, feedback
app.include_router(recommend.router, prefix="/v1")
app.include_router(feedback.router, prefix="/v1")
app.include_router(health.router, prefix="/v1")
