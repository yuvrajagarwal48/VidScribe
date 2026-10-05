"""
Session Memory Manager for VidScribe
-----------------------------------
Manages in-memory conversational dialogue history per session / video.
Enables multi-turn conversational context, pronoun resolution, and follow-up Q&A.
"""

from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Mapping from session_id -> list of BaseMessage
_SESSION_MEMORY: Dict[str, List[BaseMessage]] = {}


def get_session_messages(session_id: str) -> List[BaseMessage]:
    """Retrieve the conversation messages for a session."""
    return list(_SESSION_MEMORY.get(session_id, []))


def append_session_interaction(session_id: str, user_text: str, assistant_text: str, max_turns: int = 6) -> None:
    """Append a human question and assistant response to session memory."""
    if session_id not in _SESSION_MEMORY:
        _SESSION_MEMORY[session_id] = []
    
    _SESSION_MEMORY[session_id].append(HumanMessage(content=user_text))
    _SESSION_MEMORY[session_id].append(AIMessage(content=assistant_text))
    
    # Prune to recent turns (each turn = 2 messages: Human + AI)
    max_messages = max_turns * 2
    if len(_SESSION_MEMORY[session_id]) > max_messages:
        _SESSION_MEMORY[session_id] = _SESSION_MEMORY[session_id][-max_messages:]


def clear_session_memory(session_id: str) -> bool:
    """Clear memory for a specific session."""
    if session_id in _SESSION_MEMORY:
        del _SESSION_MEMORY[session_id]
        return True
    return False


def clear_video_sessions(video_id: str) -> int:
    """Clear all sessions associated with a specific video_id."""
    cleared = 0
    to_delete = [s for s in _SESSION_MEMORY if video_id in s]
    for s in to_delete:
        del _SESSION_MEMORY[s]
        cleared += 1
    return cleared
