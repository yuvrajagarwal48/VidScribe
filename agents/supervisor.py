"""
Supervisor Agent Node
---------------------
Analyzes incoming user intent, tracks conversational memory,
and routes execution to the appropriate specialized worker node
(IngestionAgent, RAGAgent, or MediaAgent).
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from schemas.state_models import AgentState
from core.llm_factory import LLMFactory, extract_text_from_message_content


SUPERVISOR_PROMPT = """
You are the Supervisor Orchestrator for VidScribe Video Intelligence.
Analyze the user's latest message and categorize their intent into exactly one of:

1. 'ingest': The user wants to process, index, or prepare a new video file.
2. 'summarize': The user wants an overview, general summary, highlights, or timeline of the video.
3. 'media': The user explicitly asks to generate/download a Storyboard PDF or create a summary video clip with narration.
4. 'rag_qa': The user asks a specific question about what happens in the video, when an event occurs, who is speaking, or visual details.

Respond with ONLY ONE word: 'ingest', 'summarize', 'media', or 'rag_qa'.
"""


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor node that determines the routing path based on user intent.
    """
    query = state.get("user_query", "")
    trace = list(state.get("reasoning_trace", []))
    
    # Fast heuristic shortcuts for common phrases to save an LLM round-trip
    query_lower = query.lower()
    if any(k in query_lower for k in ["pdf", "storyboard", "make video", "summary video", "montage"]):
        intent = "media"
        agent_deployed = "Media Synthesis Agent (FFmpeg & Storyboard)"
    elif any(k in query_lower for k in ["summarize", "overview", "what is this video about", "highlights", "summary"]):
        intent = "summarize"
        agent_deployed = "Media Synthesis & Narrative Agent"
    elif any(k in query_lower for k in ["process video", "index video", "ingest"]):
        intent = "ingest"
        agent_deployed = "Ingestion Agent (Whisper, PySceneDetect & OCR)"
    else:
        intent = "rag_qa"
        agent_deployed = "Multimodal RAG Agent (ChromaDB & Gemini Analyst)"

    trace.append(f"[SUPERVISOR] 🎯 Orchestrator classified intent as '{intent}' -> Deployed {agent_deployed}")
    
    return {
        "intent": intent,
        "reasoning_trace": trace
    }
