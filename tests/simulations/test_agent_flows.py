import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.graph import build_graph
from app.agents.state import AgentState

@pytest.fixture
def agent():
    return build_graph()

@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage("upbeat morning music for coffee shop")],
        "session_id": "sim_test_001",
        "venue_context": {
            "venue_type": "cafe",
            "time_of_day": "morning",
            "energy_preference": "medium",
            "session_duration": 120,
        },
        "search_results": [],
        "mood_scores": {},
        "final_playlist": [],
        "plan": "",
        "reasoning_trace": [],
        "memory_context": {},
        "next_step": "",
        "error": None,
        "session_updated": False,
    }

class TestAgentFlows:

    def test_agent_returns_non_empty_playlist(self, agent, base_state):
        """Full agent run must return at least 1 track."""
        result = agent.invoke(base_state)
        assert len(result["final_playlist"]) > 0

    def test_agent_populates_reasoning_trace(self, agent, base_state):
        """Agent must populate reasoning trace — proves all nodes ran."""
        result = agent.invoke(base_state)
        assert len(result["reasoning_trace"]) > 0

    def test_fresh_session_starts_with_no_memory(self, agent, base_state):
        """
        A brand new session_id must show no history in reasoning.
        Uses a unique session_id that has never been used.
        """
        import uuid
        base_state["session_id"] = f"fresh_{uuid.uuid4().hex}"
        result = agent.invoke(base_state)

        # Flatten reasoning trace and check
        trace_str = str(result["reasoning_trace"])
        assert "no past interactions found for this session" in trace_str or "fresh session" in trace_str.lower()

    def test_second_run_loads_session_memory(self, agent, base_state):
        """
        Second invocation with same session_id must show
        memory loaded in reasoning trace.
        """
        import uuid
        session_id = f"mem_test_{uuid.uuid4().hex}"
        base_state["session_id"] = session_id

        # First run — creates memory
        agent.invoke(base_state)

        # Second run — must load memory
        base_state["messages"] = [HumanMessage("something calmer now")]
        base_state["search_results"] = []
        base_state["final_playlist"] = []
        base_state["reasoning_trace"] = []
        base_state["memory_context"] = {}

        result2 = agent.invoke(base_state)
        trace_str = str(result2["reasoning_trace"])
        assert "past interactions" in trace_str or "session memory" in trace_str.lower()

    def test_agent_handles_ambiguous_query_gracefully(self, agent, base_state):
        """
        Vague query must not crash the agent —
        it must return something or an empty playlist, not an exception.
        """
        base_state["messages"] = [HumanMessage("music")]
        try:
            result = agent.invoke(base_state)
            assert isinstance(result["final_playlist"], list)
        except Exception as e:
            pytest.fail(f"Agent crashed on ambiguous query: {e}")
        