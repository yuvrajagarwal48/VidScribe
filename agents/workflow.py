"""
LangGraph Multi-Agent Workflow
------------------------------
Assembles the complete state machine graph:
1. Supervisor routes incoming queries.
2. IngestionAgent processes new videos via progressive indexing.
3. RAGAgent performs multimodal retrieval with on-demand keyframe inspection.
4. MediaAgent synthesizes summaries, storyboard PDFs, and video highlights.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from schemas.state_models import AgentState
from agents.supervisor import supervisor_node
from agents.ingestion_agent import ingestion_node
from agents.rag_agent import rag_node
from agents.media_agent import media_node


def route_supervisor(state: AgentState) -> str:
    """Evaluates the state's classified intent and returns the next node key."""
    intent = state.get("intent", "rag_qa")
    if intent == "ingest":
        return "ingestion"
    elif intent in ["summarize", "media"]:
        return "media"
    else:
        return "rag"


def build_videorag_graph():
    """
    Constructs and compiles the stateful multi-agent LangGraph workflow.
    """
    # 1. Initialize StateGraph with our Pydantic-compatible TypedDict state
    builder = StateGraph(AgentState)

    # 2. Add processing nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("rag", rag_node)
    builder.add_node("media", media_node)

    # 3. Define Entrypoint
    builder.set_entry_point("supervisor")

    # 4. Add conditional routing edges
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "ingestion": "ingestion",
            "rag": "rag",
            "media": "media"
        }
    )

    # 5. Worker nodes finish execution at END
    builder.add_edge("ingestion", END)
    builder.add_edge("rag", END)
    builder.add_edge("media", END)

    # 6. Attach in-memory checkpointer for multi-turn conversational session memory
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    
    return app


# Compiled workflow instance ready for application execution
videorag_agent_app = build_videorag_graph()
