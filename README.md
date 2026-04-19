# TuneHub
An AI agent that takes a natural language query like "find me upbeat indie tracks for a coffee shop playlist" and autonomously uses tools to search, filter, rank, and return music recommendations with explanations.

## 🎵 Features
* **Context-Aware Recommendations**: Tailors playlists to venue types (cafe, gym, spa), time of day, and energy preferences.
* **Autonomous AI Agent**: Uses a LangGraph-powered orchestrator to plan, execute, and synthesize music curation workflows.
* **Tool-Augmented Generation**: Incorporates custom tools like `music_search`, `mood_analyzer`, and `playlist_builder` to fetch and curate tracks.
* **Dual Memory System**:
  * **Short-term Memory (Session)**: Uses Redis to track conversation history, allowing for context-aware refinements.
  * **Long-term Memory (Semantic)**: Integrates ChromaDB to retrieve embeddings of similar, previously successful playlists.
* **REST API**: Built with FastAPI, ready to connect with frontends.
* **Containerized**: Clean setup and seamless execution using Docker Compose.

## 💻 Tech Stack
* **Python Framework**: FastAPI
* **Agentic Framework**: LangGraph, LangChain
* **LLM Provider**: Groq (Llama-3.3-70b-versatile), Google Generative AI (Gemini 2.0 Flash)
* **Vector Database**: ChromaDB + FastEmbed
* **Cache/Memory Store**: Redis
* **Package Manager**: uv

## 🚀 Getting Started

### Prerequisites
* Python >= 3.12
* [uv](https://github.com/astral-sh/uv) (for ultra-fast dependency management)
* Docker & Docker Compose (Optional, for containerized run)
* Redis (if running locally without Docker)

### 1. Local Setup

Navigate to the project root and install dependencies using `uv`:

```bash
# Install dependencies
uv sync
```

Set up your environment variables. create a `.env` file based on `.env.example` in the project root:

```bash
cp .env.example .env
```

Populate the `.env` file with your API keys:
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `LANGSMITH_API_KEY` (if using tracing)

Start the local Redis server:
```bash
redis-server
```

Run the FastAPI application:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
or run the local simulation script:
```bash
uv run python main.py
```

### 2. Docker Setup

You can run the entire application, along with Redis, using Docker Compose:

```bash
cd docker
docker-compose up -d --build
```
The application will be accessible at `http://localhost:8080`.

## 🧠 Application Architecture (LangGraph)
TuneHub runs through a sophisticated cognitive architecture using a LangGraph `StateGraph`:
1. **Memory Load Node**: Retrieves session history (Redis) and semantically similar past playlists (ChromaDB).
2. **Planner Node**: Instructs the LLM to analyze the venue's context and user query to derive search context.
3. **Tool Node**: Executes necessary tool iterations (`music_search_tool`, `mood_analyzer_tool`, `playlist_builder_tool`).
4. **Synthesizer Node**: Evaluates the findings and shortlists the final 5-8 tracks along with a curation rationale.
5. **Memory Save Node**: Commits the new playlist to ChromaDB's vector store and updates the session history in Redis.

## 🔗 API Endpoints
* `POST /v1/recommend`: Receives context and returns an AI-curated playlist.
* `POST /v1/feedback`: Extends memory by processing user feedback on recommended playlists.
* `GET /v1/health`: Returns API health status.

Check `http://localhost:8080/docs` while the app runs for the interactive Swagger UI and complete API documentation.
