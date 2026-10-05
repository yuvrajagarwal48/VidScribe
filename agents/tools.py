"""
LangChain Agent Tools
---------------------
Decoupled, strongly-typed LangChain tools (@tool) providing the agent
access to ChromaDB retrieval, on-demand keyframe inspection, OCR, and media generation.
"""

import json
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

from core.vector_db import VectorDBStore
from core.vision_engine import VisionEngine
from core.audio_engine import AudioEngine
from core.media_synthesis import MediaSynthesisEngine

# Instantiate reusable engine instances
_vector_store = VectorDBStore()
_vision_engine = VisionEngine()
_audio_engine: Optional[AudioEngine] = None
_media_engine = MediaSynthesisEngine()


def _get_audio() -> AudioEngine:
    global _audio_engine
    if _audio_engine is None:
        _audio_engine = AudioEngine()
    return _audio_engine


@tool
def search_video_context(query: str, video_id: Optional[str] = None) -> str:
    """
    Search indexed video scenes (dialogue transcripts, on-screen OCR text, visual elements)
    in the ChromaDB vector database using semantic similarity.
    
    Args:
        query: The semantic search text or question
        video_id: Optional video ID to limit search to a single video
        
    Returns:
        JSON string of matched scenes with start/end timestamps and descriptions
    """
    hits = _vector_store.query(query_text=query, video_id=video_id, top_k=6)
    formatted = []
    for hit in hits:
        meta = hit.get("metadata", {})
        formatted.append({
            "scene_index": meta.get("scene_index"),
            "formatted_start": meta.get("formatted_start"),
            "formatted_end": meta.get("formatted_end"),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "type": meta.get("type"),
            "content": hit.get("document", ""),
            "keyframe_path": meta.get("keyframe_path")
        })
    return json.dumps(formatted, indent=2)


@tool
def inspect_keyframe(image_path: str, question: Optional[str] = None) -> str:
    """
    Inspect a specific video keyframe image using Gemini Vision to answer
    visual questions (e.g. colors, clothing, objects, gestures, slide graphics).
    
    Args:
        image_path: Path to the keyframe JPEG image
        question: Specific visual question to inspect in the frame
        
    Returns:
        JSON string containing detected objects and scene description
    """
    res = _vision_engine.inspect_frame_visuals(image_path, user_prompt=question)
    return json.dumps(res, indent=2)


@tool
def run_ocr_on_frame(image_path: str) -> str:
    """
    Runs Tesseract OCR on a keyframe to extract text on slides, signs, shirts, or scoreboards.
    
    Args:
        image_path: Path to keyframe JPEG image
        
    Returns:
        Comma-separated string of detected on-screen words
    """
    tokens = _vision_engine.perform_ocr(image_path)
    words = [t.text for t in tokens]
    return ", ".join(words) if words else "No text detected in this frame."


@tool
def transcribe_dialogue(video_path: str, start_time: float, end_time: float) -> str:
    """
    Runs Whisper speech recognition on a specific slice of the video audio track.
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Transcribed spoken dialogue text
    """
    audio = _get_audio()
    segments = audio.transcribe_segment(video_path, start_time, end_time)
    lines = [f"[{s.start_time:.1f}s - {s.end_time:.1f}s]: {s.transcript}" for s in segments]
    return "\n".join(lines) if lines else "No spoken dialogue detected in this time range."
