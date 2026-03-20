from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from app.agents.state import AgentState
from app.tools.music_search import music_search_tool
from app.config import get_settings

settings = get_settings()

#---LLM---
llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.1,
    api_key=settings.groq_api_key
)

#bind tools
llm_with_tools = llm.bind_tools([music_search_tool])

#Node 1: Planner

@traceable("tool")
def planner_node(state: AgentState) -> dict:
    """LLM decide What to do based on Query."""
    
    system_prompt = """You are Music Mind, an expert music curator for business venues.
    
    Given a query and venue context, use the music_search_tool to find appropriate tracks.
    Always call the tool — never respond without searching first.
    
    Think step by step.
    1. What genre fits this venue and time of day?
    2. What energy level is appropriate?
    3. What BPM range makes sense?
    """

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "plan": "Searching music catalog based on venue context.",
        "resoning_trace": [f"Planner called LLM: {response.content[:100]}"]
    }

#Node2: Search Tool Node
@traceable("chain")
def tool_node(state: AgentState) -> dict:
    """Execute whatever tools the planner requested."""
    from langgraph.prebuilt import ToolNode

    last_message = state["messages"][-1]
    tool_result = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "music_search_tool":
            result = music_search_tool.invoke(tool_call["args"])
            tool_result.extend(result)

    state["search_results"] = state["search_results"] + tool_result
    state["reasoning_trace"] = state["reasoning_trace"].extend([f"Tools returned {len(tool_result)} results."])
    return {
        "search_results": state["search_results"],
        "resoning_trace": state["reasoning_trace"]
    }
    

#Node3: Synthesizer Node
@traceable("llm")
def synthesizer_node(state: AgentState) -> dict:
    """Turn raw tool results into a final playlist."""
    
    track = state["search_results"][:10]

    synthesis_prompt = f"""
    You found {len(track)} tracks.
    Venue Context: {state['venue_context']['venue_type']}
    Time of Day: {state['venue_context']['time_of_day']}

    Select the best 5-8 tracks and explain why they are a good fit this venue.
    The Format should be:
    - ID: "unique_track_id"
    - Title: "track title"
    - Artist: "artist name"
    - BPM: "bpm value"
    Return your response as a brief curation rationale followed by the track list.
    """

    response = llm.invoke([HumanMessage(synthesis_prompt)])

    return {
        "final_playlist": track[:8],
        "messages": [response],
        "reasoning_trace": state["reasoning_trace"] + ["Synthesizer completed."]
    }

#---Routing Logic---
@traceable("tool")
def should_use_tools(state: AgentState) -> str:
    """check if last message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"
    return "synthesizer"

#---Build Graph---

def build_graph():
    graph = StateGraph(AgentState)

    #Add Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesizer", synthesizer_node)

    #Add Edges
    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        should_use_tools,
        {
            "tool": "tool",
            "synthesizer": "synthesizer"
        }
    )
    graph.add_edge("tool", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()

#Singleton - Compile graph once and reuse
agent = build_graph()