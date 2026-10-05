"""
Pydantic Schemas Package
-----------------------
Defines structured data contracts across video processing, RAG, and agent state.
"""
from schemas.video_models import (
    SceneBoundary,
    KeyframeData,
    TranscriptSegment,
    OCRToken,
    DetectedObject,
    ObjectSpatialRelation,
    SceneAnalysisResult,
)
from schemas.rag_models import (
    GroundedSegment,
    QueryRequest,
    QueryResponse,
    VideoSummaryResult,
)
from schemas.state_models import AgentState

__all__ = [
    "SceneBoundary",
    "KeyframeData",
    "TranscriptSegment",
    "OCRToken",
    "DetectedObject",
    "ObjectSpatialRelation",
    "SceneAnalysisResult",
    "GroundedSegment",
    "QueryRequest",
    "QueryResponse",
    "VideoSummaryResult",
    "AgentState",
]
