"""
Agent State Models
------------------
Defines the shared state dictionary and typing for the LangGraph multi-agent workflow.
"""

from typing import Annotated, List, Dict, Any, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Central state tracked across all nodes in the LangGraph workflow.
    `messages` uses `add_messages` reducer to accumulate dialogue history.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    video_id: Optional[str]
    video_path: Optional[str]
    user_query: str
    
    # Intent category determined by Supervisor: 'qa', 'summarize', 'find_timestamp', 'render_media'
    intent: str
    
    # Retrieved evidence from ChromaDB and perception tools
    retrieved_segments: List[Dict[str, Any]]
    
    # On-demand inspected keyframe paths
    active_keyframes: List[str]
    
    # Self-reflective iteration counter (prevents infinite loops in CRAG)
    reflection_count: int
    is_sufficient: bool
    
    # Generated outputs
    final_answer: Optional[str]
    grounded_segments: List[Dict[str, Any]]
    reasoning_trace: List[str]
    
    # Media artifacts (e.g. summary video, storyboard PDF)
    summary_data: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]
