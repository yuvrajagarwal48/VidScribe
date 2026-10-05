"""
RAG and Query Data Models
-------------------------
Models for user queries, retrieval grounding, grounded timestamp segments,
and full-video summaries with storyboards.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GroundedSegment(BaseModel):
    """Represents a specific video segment cited as evidence for an answer."""
    video_id: str
    scene_index: int
    start_time: float
    end_time: float
    formatted_start: str
    formatted_end: str
    modality_type: str = Field(default="combined", description="'ocr', 'asr', 'object', or 'combined'")
    source_preview: Optional[str] = Field(default=None, description="Keyframe image path or text snippet")


class QueryRequest(BaseModel):
    """Payload for natural language video querying."""
    query: str = Field(..., description="User's natural language question or request")
    video_id: Optional[str] = Field(default=None, description="Optional video ID to filter search scope")
    session_id: Optional[str] = Field(default=None, description="Optional session or chat thread ID for multi-turn conversational memory")
    top_k: int = Field(default=8, description="Number of candidate scenes to retrieve")
    stream: bool = Field(default=False, description="Whether to stream the generated answer")


class QueryResponse(BaseModel):
    """Structured response containing the grounded answer, citations, and thought trace."""
    query: str
    response: str = Field(..., description="LLM generated answer grounded in video evidence")
    video_segments: List[GroundedSegment] = Field(default_factory=list, description="Citations used for grounding")
    reasoning_steps: List[str] = Field(default_factory=list, description="Agent execution trace / thought log")


class KeySceneHighlight(BaseModel):
    """Specific key scene selected for a summary video montage."""
    scene_index: int
    timestamp_range: str
    importance_reason: str
    keyframe_path: Optional[str] = None


class VideoSummaryResult(BaseModel):
    """Full-video executive summary, highlights timeline, and exported media paths."""
    video_id: str
    summary_text: str = Field(..., description="Concise overall summary (1-2 paragraphs)")
    key_points: List[str] = Field(default_factory=list, description="Bullet-point highlights")
    timeline_events: List[str] = Field(default_factory=list, description="Chronological timeline of events")
    key_scenes: List[KeySceneHighlight] = Field(default_factory=list)
    summary_video_path: Optional[str] = Field(default=None, description="Path to generated summary MP4 clip")
    storyboard_pdf_path: Optional[str] = Field(default=None, description="Path to generated Storyboard PDF")
