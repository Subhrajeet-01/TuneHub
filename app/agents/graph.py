from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from app.agents.state import AgentState
from app.tools.music_search import music_search_tool
from app.tools.mood_analyser import mood_analyzer_tool
from app.tools.playlist_builder import playlist_builder_tool
from app.memory.session import SessionMemory
from app.memory.semantic import SemanticMemory
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
def planner_node(state: AgentState) -> dict:
    """LLM decide What to do based on Query."""
    memory = state.get("memory_context", {})
    preferences = memory.get("preferences", {})
    history = memory.get("history", [])

    # Build Memory context string for prompt
    memory_str = ""
    if preferences:
        memory_str += f"\nKnown Preferences: {preferences}"
    if history:
        last = history[-1]
        memory_str += f"\nLast Query was: '{last.get('query', '')}'"
        memory_str += f"\nLast playlist included tracks: {last.get('playlist_ids', [])}"
    
    system_prompt = f"""You are Music Mind, an expert music curator for business venues.
    
    IMPORTANT: You MUST always call the music_search_tool before responding.
    Never generate a playlist from memory alone — always search first.
    Even for follow-up or refinement queries, always call the tool with updated parameters.
    {memory_str}
    
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
def tool_node(state: AgentState) -> dict:
    """Execute whatever tools the planner requested."""
    from langgraph.prebuilt import ToolNode

    last_message = state["messages"][-1]
    tool_result = []

    #step 1
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "music_search_tool":
            result = music_search_tool.invoke(tool_call["args"])
            tool_result.extend(result)

    state["search_results"] = state["search_results"] + tool_result

    #step 2
    query_text = state["messages"][0].content
    mood_results = mood_analyzer_tool.invoke({
        "tracks": state["search_results"],
        "target_mood": query_text
    })

    #step 3
    final_playlist = playlist_builder_tool.invoke({
        "tracks": mood_results,
        "venue_type": state["venue_context"]["venue_type"],
        "energy_preference": state["venue_context"]["energy_preference"],
        "limit": 8
    })

    state["reasoning_trace"] = state["reasoning_trace"].extend([
        f"Search: {len(tool_result)} tracks -> "
        f"Mood Filter -> "
        f"Playlist: {len(final_playlist)} tracks"
        ])

    return {
        "search_results": state["search_results"],
        "final_playlist": final_playlist,
        "resoning_trace": state["reasoning_trace"] 
    }
    

#Node3: Synthesizer Node
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

#Node 4: Memory Node
def memory_load_node(state: AgentState) -> dict:
    """
    First node - runs before planning.
    Loads session history from redis and injects into state for context.
    """
    session_id = state["session_id"]
    query = state["messages"][0].content
    venue_type = state["venue_context"]["venue_type"]

    # Redis
    session_context = SessionMemory.get(session_id)

    #chromadb
    similar_playlists = SemanticMemory.retrieve_similar(
        query=query,
        venue_type=venue_type,
        top_k=3
    )
    
    reasoning = []
    if session_context["history"]:
        reasoning.append(
            f"Redis: {len(session_context['history'])} past interactions loaded"
        )
    else:
        reasoning.append("Redis: no past interactions found for this session")

    if similar_playlists:
        reasoning.append(
            f"ChromaDB: {len(similar_playlists)} similar playlists retrieved"
        )
    else:
        reasoning.append("ChromaDB: no similar playlists found yet")

    return {
        "memory_context": {
            **session_context,
            "similar_playlists": similar_playlists
        },
        "reasoning_trace": state["reasoning_trace"] + [reasoning]
    }

#Node 5: Memory Save Node
def memory_save_node(state: AgentState) -> dict:
    """
    Last node - runs after synthesizer.
    Saves relevant context back to Redis for future sessions.
    Save the similar playlist as an embedding in chromadb for future retrieval.
    """ 
    import uuid

    session_id = state["session_id"]
    track_ids = [t["id"] for t in state["final_playlist"]]

    # save to Redis
    interaction = {
        "query": state["messages"][0].content,
        "venue_type": state["venue_context"]["venue_type"],
        "playlist_ids": [track["id"] for track in state["final_playlist"]],
        "preferences": {
            "energy": state["venue_context"]["energy_preference"],
            "venue_type": state["venue_context"]["venue_type"]
        }
    }

    SessionMemory.update(session_id, interaction)

    # save to ChromaDB
    SemanticMemory.store_playlist(
        playlist_id = f"pl_{uuid.uuid4().hex[:8]}",
        query = state["messages"][0].content,
        venue_type = state["venue_context"]["venue_type"],
        track_ids = track_ids,
        energy = state["venue_context"]["energy_preference"]
    )

    return {
        
        "reasoning_trace": state["reasoning_trace"] + [
            f"Memory saved to Redis: {SessionMemory.count()} interactions stored.",
            f"ChromaDB updated with new playlist embedding. Total stored Playlists: {SemanticMemory.count()}"
        ]
    }

#---Routing Logic---
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
    graph.add_node("memory_load", memory_load_node)
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("memory_save", memory_save_node)

    #Add Edges
    graph.set_entry_point("memory_load")
    graph.add_edge("memory_load", "planner")
    graph.add_conditional_edges(
        "planner",
        should_use_tools,
        {
            "tool": "tool",
            "synthesizer": "synthesizer"
        }
    )
    graph.add_edge("tool", "synthesizer")
    graph.add_edge("synthesizer", "memory_save")
    graph.add_edge("memory_save", END)

    return graph.compile()

#Singleton - Compile graph once and reuse
agent = build_graph()