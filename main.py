# delete this after testing, it's just for smoke testing

from app.agents.graph import build_graph
from app.agents.state import AgentState
from langchain_core.messages import HumanMessage

initial_state: AgentState = {
    "messages": [HumanMessage(content="intense workout music for a gym")],
    "session_id": "test_session_001",
    "venue_context": {
        "venue_type": "gym",
        "time_of_day": "afternoon",
        "energy_preference": "high",
        "session_duration": 120
    },
    "search_results": [],
    "mood_scores": {},
    "final_playlist": [],
    "plan": "",
    "reasoning_trace": [],
    "memory_context": [],
    "next_step": "",
    "error": None
}
agent = build_graph()
result = agent.invoke(initial_state)
print(result["messages"][-1].content)


if __name__ == "__main__":
    pass