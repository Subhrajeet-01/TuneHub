"""
Eval Playlist Agent using llm.
Run separately: pytest tests/evals/ -v
"""
import pytest
import csv
import os
from datetime import datetime
from app.agents.graph import build_graph
from app.agents.state import AgentState
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from app.config import get_settings

settings = get_settings()

agent = build_graph()
judge_llm = ChatGroq(model=settings.groq_model)

EVAL_SCENARIOS = [
    {
        "query": "upbeat morning music for coffee shop",
        "venue_type": "cafe",
        "energy": "medium",
        "expected_mood": "upbeat and cheerful",
    },
    {
        "query": "intense workout playlist for gym",
        "venue_type": "gym",
        "energy": "high",
        "expected_mood": "high energy and driving",
    },
    {
        "query": "relaxing spa background music",
        "venue_type": "spa",
        "energy": "low",
        "expected_mood": "calm and peaceful",
    },
]

def judge_playlist(query, venue_type, tracks, expected_mood) -> dict:
    """Ask LLM to score the playlist on helpfulness and mood match."""
    track_list = "\n".join(
        f"- {t['title']} by {t['artist']} (BPM: {t['bpm']}, energy: {t['energy']})"
        for t in tracks
    )

    prompt = f"""You are evaluating a music recommendation system for B2B venues.

Query: "{query}"
Venue: {venue_type}
Expected mood: {expected_mood}

Recommended playlist:
{track_list}

Score this playlist on each dimension from 1-5:
1. Helpfulness: Does it match the query intent?
2. Mood match: Does it match the expected mood ({expected_mood})?
3. Venue fit: Is it appropriate for a {venue_type}?

Respond ONLY in this exact format:
helpfulness: X
mood_match: X
venue_fit: X
reasoning: one sentence explanation
"""
    response = judge_llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    scores = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            scores[key.strip()] = val.strip()
    return scores

def save_eval_results(results: list[dict]):
    """Save eval results to CSV for tracking over time."""
    os.makedirs("experiments", exist_ok=True)
    filename = f"experiments/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Eval results saved to {filename}")


class TestPlaylistAgent:

    def test_eval_all_scenarios(self):
        """Run LLM-as-jugde eval on all scenarios and log results."""
        import uuid
        all_results = []

        for scenario in EVAL_SCENARIOS:
            state: AgentState = {
                "messages": [HumanMessage(scenario["query"])],
                "session_id": f"eval_{uuid.uuid4().hex[:6]}",
                "venue_context": {
                    "venue_type": scenario["venue_type"],
                    "time_of_day": "morning",
                    "energy_preference": scenario["energy"],
                    "session_duration": 60,
                },
                "search_results": [], "mood_scores": {},
                "final_playlist": [], "plan": "",
                "reasoning_trace": [], "memory_context": {},
                "next_step": "", "error": None, "session_updated": False,
            }

            result = agent.invoke(state)
            scores = judge_playlist(
                query=scenario["query"],
                venue_type=scenario["venue_type"],
                tracks=result["final_playlist"],
                expected_mood=scenario["expected_mood"]
            )

            row = {
                "query": scenario["query"],
                "venue_type": scenario["venue_type"],
                **scores,
                "track_count": len(result["final_playlist"]),
            }
            all_results.append(row)

            print(f"\n{'='*50}")
            print(f"Query: {scenario['query']}")
            print(f"Scores: {scores}")

            # Assert minimum quality bar
            helpfulness = int(scores.get("helpfulness", 0))
            assert helpfulness >= 3, \
                f"Helpfulness score {helpfulness} below threshold for: {scenario['query']}"

        save_eval_results(all_results)