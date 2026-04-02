
from app.agents.graph import build_graph
from app.agents.state import AgentState
from langchain_core.messages import HumanMessage

queries = [
    ("upbeat morning music for cafe",       "cafe",  "morning",   "medium"),
    ("energetic gym workout playlist",      "gym",   "afternoon", "high"),
    ("calm spa relaxation music",           "spa",   "evening",   "low"),
    ("cheerful coffee shop morning vibes",   "cafe",  "morning",   "medium"),  # This last one should retrieve the first query from ChromaDB
]

agent = build_graph()
for i, (query, venue, time, energy) in enumerate(queries):
    state: AgentState = {
        "messages": [HumanMessage(query)],
        "session_id": f"test_session_{i}",
        "venue_context": {
            "venue_type": venue,
            "time_of_day": time,
            "energy_preference": energy,
            "session_duration": 120,
        },
        "search_results": [], "mood_scores": {}, "final_playlist": [],
        "plan": "", "reasoning_trace": [], "memory_context": {},
        "next_step": "", "error": None, "session_updated": False
    }

    result = agent.invoke(state)
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"Reasoning trace:")
    for step in result["reasoning_trace"]:
        print(f"  → {step}")
    print(f"Playlist: {[t['title'] for t in result['final_playlist']]}")

if __name__ == "__main__":
    pass