"""
Video Data Models
-----------------
Strongly-typed Pydantic v2 data models for video scenes, transcripts,
OCR tokens, objects, and complete multimodal frame analysis.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SceneBoundary(BaseModel):
    """Represents a temporal cut / scene detected in the video."""
    scene_index: int = Field(..., description="0-indexed scene number")
    start_frame: int = Field(..., description="Starting frame number")
    end_frame: int = Field(..., description="Ending frame number")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    formatted_start: str = Field(..., description="Formatted start timestamp (HH:MM:SS.mmm)")
    formatted_end: str = Field(..., description="Formatted end timestamp (HH:MM:SS.mmm)")
    duration: float = Field(..., description="Duration of scene in seconds")


class KeyframeData(BaseModel):
    """Represents an extracted keyframe snapshot for a scene."""
    scene_index: int
    image_path: str = Field(..., description="Absolute or relative path to saved JPG keyframe")
    middle_frame: int
    middle_time: float = Field(..., description="Timestamp of keyframe in seconds")
    formatted_middle: str = Field(..., description="Formatted timestamp (HH:MM:SS.mmm)")


class TranscriptSegment(BaseModel):
    """Represents an audio segment transcribed by Whisper."""
    transcript: str = Field(..., description="Transcribed spoken dialogue text")
    start_time: float = Field(default=0.0, description="Segment start in seconds")
    end_time: float = Field(default=0.0, description="Segment end in seconds")
    confidence: float = Field(default=1.0, description="Transcription confidence score")


class OCRToken(BaseModel):
    """Represents a word or text element extracted from a frame via OCR."""
    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., description="OCR confidence (0 - 100)")
    x: int = Field(..., description="Bounding box X coordinate")
    y: int = Field(..., description="Bounding box Y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")


class ObjectSpatialRelation(BaseModel):
    """Represents a spatial or functional relationship between detected objects."""
    target: int = Field(..., description="Index of target object in detected list")
    relation: str = Field(..., description="Predicate such as 'next_to', 'above', 'inside'")


class DetectedObject(BaseModel):
    """Represents a visual object detected by Gemini Vision or another multimodal model."""
    class_name: str = Field(..., description="Object category, e.g. 'person', 'car', 'laptop'")
    bbox: List[float] = Field(default_factory=list, description="Coordinates [x1, y1, x2, y2] as percentages")
    attributes: List[str] = Field(default_factory=list, description="Descriptive attributes, e.g. 'red', 'metallic'")
    relationships: List[ObjectSpatialRelation] = Field(default_factory=list, description="Spatial relations to other objects")


class SceneAnalysisResult(BaseModel):
    """Unified multimodal representation of a single video scene."""
    video_id: str
    scene_index: int
    keyframe_path: str
    start_time: float
    end_time: float
    formatted_start: str
    formatted_end: str
    ocr_texts: List[OCRToken] = Field(default_factory=list)
    transcripts: List[TranscriptSegment] = Field(default_factory=list)
    detected_objects: List[DetectedObject] = Field(default_factory=list)
    summary_text: Optional[str] = Field(default=None, description="Concise textual summary of the scene")
