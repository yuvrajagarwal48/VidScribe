"""
FastAPI Route Handlers
----------------------
Defines REST API endpoints for video management, progressive processing,
multimodal query execution, and media downloading.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
from schemas.rag_models import QueryRequest, QueryResponse, GroundedSegment
from agents.workflow import videorag_agent_app
from core.vector_db import VectorDBStore
from core.video_engine import VideoEngine
from backend.session_memory import (
    get_session_messages,
    append_session_interaction,
    clear_session_memory,
    clear_video_sessions
)

router = APIRouter(prefix="/api")
vector_store = VectorDBStore()


# =========================================================================
# 1. Video Upload & Listing Endpoints
# =========================================================================

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a new video file to the data repository.
    Generates a fresh session ID for a new dedicated chat window.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Clean filename
    clean_name = Path(file.filename).name
    save_path = config.DATA_DIR / clean_name
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_id = save_path.stem
    session_id = f"session_{video_id}_{int(save_path.stat().st_mtime)}"
    return {
        "status": "success",
        "video_id": video_id,
        "filename": clean_name,
        "video_path": str(save_path),
        "session_id": session_id
    }


@router.get("/videos")
async def list_videos():
    """
    Lists all available videos in the data directory and their indexing status.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []

    for f in config.DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            vid_id = f.stem
            # Check if this video has scenes in the vector store
            existing_scenes = vector_store.get_all_scenes_for_video(vid_id)
            
            # Check if summary artifacts exist
            pdf_path = config.SUMMARIES_DIR / f"{vid_id}_storyboard.pdf"
            summary_vid = config.SUMMARIES_DIR / f"{vid_id}_summary.mp4"

            videos.append({
                "video_id": vid_id,
                "filename": f.name,
                "file_size": f.stat().st_size,
                "indexed": len(existing_scenes) > 0,
                "scene_count": len(existing_scenes),
                "has_storyboard": pdf_path.exists(),
                "has_summary_video": summary_vid.exists(),
                "video_url": f"/api/media/video/{f.name}"
            })

    return {"videos": videos}


# =========================================================================
# 2. Timeline & Video Inspection Endpoints
# =========================================================================

@router.get("/videos/{video_id}/timeline")
async def get_video_timeline(video_id: str):
    """
    Returns the scene timeline, timestamps, and keyframe image URLs for a video.
    """
    scenes = vector_store.get_all_scenes_for_video(video_id)
    if not scenes:
        # Check if video exists locally but isn't indexed yet
        vid_path = None
        for f in config.DATA_DIR.iterdir():
            if f.stem == video_id:
                vid_path = str(f)
                break
        
        if vid_path:
            return {"video_id": video_id, "indexed": False, "scenes": []}
        raise HTTPException(status_code=404, detail="Video not found")

    timeline = []
    for s in scenes:
        meta = s.get("metadata", {})
        idx = meta.get("scene_index")
        kf_filename = f"{video_id}_scene_{idx:03d}.jpg"
        
        timeline.append({
            "scene_index": idx,
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "formatted_start": meta.get("formatted_start", "00:00"),
            "formatted_end": meta.get("formatted_end", "00:00"),
            "description": s.get("document", ""),
            "keyframe_url": f"/api/media/keyframe/{video_id}/{kf_filename}"
        })

    return {"video_id": video_id, "indexed": True, "scenes": timeline}


# =========================================================================
# 3. Progressive Ingestion Endpoint
# =========================================================================

@router.post("/process/{video_id}")
async def process_video(video_id: str):
    """
    Triggers progressive Tier 1 indexing via the IngestionAgent.
    """
    # Locate original video file
    video_path = None
    for f in config.DATA_DIR.iterdir():
        if f.stem == video_id:
            video_path = str(f)
            break

    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video file for '{video_id}' not found.")

    # Invoke LangGraph Ingestion Node
    init_state = {
        "messages": [],
        "video_id": video_id,
        "video_path": video_path,
        "user_query": f"process video {video_path}",
        "intent": "ingest",
        "retrieved_segments": [],
        "active_keyframes": [],
        "reflection_count": 0,
        "is_sufficient": True,
        "final_answer": None,
        "grounded_segments": [],
        "reasoning_trace": [],
        "summary_data": None,
        "artifacts": {}
    }

    # Execute workflow with unique thread ID for session isolation
    config_dict = {"configurable": {"thread_id": f"ingest_{video_id}"}}
    result = videorag_agent_app.invoke(init_state, config_dict)

    return {
        "status": "success",
        "video_id": video_id,
        "message": result.get("final_answer"),
        "reasoning_trace": result.get("reasoning_trace", [])
    }


# =========================================================================
# 4. Agentic RAG Query Endpoint
# =========================================================================

@router.post("/query", response_model=QueryResponse)
async def query_video(req: QueryRequest):
    """
    Executes natural language RAG query through the LangGraph StateGraph.
    Maintains short-term conversational session memory across multi-turn queries.
    Returns grounded answer with exact timestamp citations and reasoning trace.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = req.session_id or f"session_{req.video_id or 'global'}"

    # Locate video file if video_id is provided
    video_path = None
    if req.video_id:
        for f in config.DATA_DIR.iterdir():
            if f.stem == req.video_id:
                video_path = str(f)
                break

    # Retrieve prior conversation turns from session memory
    prior_messages = get_session_messages(session_id)

    init_state = {
        "messages": prior_messages,
        "video_id": req.video_id,
        "video_path": video_path,
        "user_query": req.query,
        "intent": "rag_qa",
        "retrieved_segments": [],
        "active_keyframes": [],
        "reflection_count": 0,
        "is_sufficient": True,
        "final_answer": None,
        "grounded_segments": [],
        "reasoning_trace": [],
        "summary_data": None,
        "artifacts": {}
    }

    config_dict = {"configurable": {"thread_id": f"query_{session_id}"}}
    res = videorag_agent_app.invoke(init_state, config_dict)

    final_answer = res.get("final_answer") or "No answer could be generated."

    # Persist conversational memory turn
    append_session_interaction(session_id, req.query, final_answer)

    segments = []
    for seg in res.get("grounded_segments", []):
        segments.append(GroundedSegment(**seg))

    return QueryResponse(
        query=req.query,
        response=final_answer,
        video_segments=segments,
        reasoning_steps=res.get("reasoning_trace", [])
    )


# =========================================================================
# 4b. Chat Session Management Endpoints
# =========================================================================

@router.delete("/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    """Resets memory for a specific conversational session."""
    cleared = clear_session_memory(session_id)
    return {"status": "success", "session_id": session_id, "cleared": cleared}


@router.delete("/chat/video/{video_id}")
async def clear_video_chat_memory(video_id: str):
    """Resets all chat sessions associated with a specific video ID."""
    count = clear_video_sessions(video_id)
    return {"status": "success", "video_id": video_id, "sessions_cleared": count}


# =========================================================================
# 5. Full Video Summarization & Export Endpoint
# =========================================================================

@router.post("/summarize/{video_id}")
async def summarize_video(video_id: str, create_video_clip: bool = False):
    """
    Generates a full executive summary, highlights timeline,
    exportable PDF storyboard, and optional summary video.
    """
    video_path = None
    for f in config.DATA_DIR.iterdir():
        if f.stem == video_id:
            video_path = str(f)
            break

    query_intent = "create video montage and summary" if create_video_clip else "summarize video and pdf"
    init_state = {
        "messages": [],
        "video_id": video_id,
        "video_path": video_path,
        "user_query": query_intent,
        "intent": "summarize",
        "retrieved_segments": [],
        "active_keyframes": [],
        "reflection_count": 0,
        "is_sufficient": True,
        "final_answer": None,
        "grounded_segments": [],
        "reasoning_trace": [],
        "summary_data": None,
        "artifacts": {}
    }

    config_dict = {"configurable": {"thread_id": f"summary_{video_id}"}}
    res = videorag_agent_app.invoke(init_state, config_dict)

    artifacts = res.get("artifacts", {})
    return {
        "video_id": video_id,
        "summary": res.get("final_answer"),
        "storyboard_pdf_url": f"/api/media/storyboard/{video_id}" if "storyboard_pdf" in artifacts else None,
        "summary_video_url": f"/api/media/summary-video/{video_id}" if "summary_video" in artifacts else None,
        "reasoning_trace": res.get("reasoning_trace", [])
    }


# =========================================================================
# 6. Static Media Serving Endpoints
# =========================================================================

@router.get("/media/video/{filename}")
async def stream_raw_video(filename: str):
    """Streams the raw video file to the HTML5 video player."""
    vid_file = config.DATA_DIR / filename
    if not vid_file.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=str(vid_file), media_type="video/mp4")


@router.get("/media/keyframe/{video_id}/{filename}")
async def get_keyframe_image(video_id: str, filename: str):
    """Serves extracted JPEG keyframes for the timeline and scene cards."""
    img_path = config.KEYFRAMES_DIR / video_id / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Keyframe image not found")
    return FileResponse(path=str(img_path), media_type="image/jpeg")


@router.get("/media/storyboard/{video_id}")
async def download_storyboard_pdf(video_id: str):
    """Streams the generated Storyboard PDF for preview or download."""
    pdf_path = config.SUMMARIES_DIR / f"{video_id}_storyboard.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Storyboard PDF not found. Generate it first.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{video_id}_storyboard.pdf"
    )


@router.get("/media/summary-video/{video_id}")
async def stream_summary_video(video_id: str):
    """Streams the compiled highlight video montage with AI voiceover."""
    mp4_path = config.SUMMARIES_DIR / f"{video_id}_summary.mp4"
    if not mp4_path.exists():
        raise HTTPException(status_code=404, detail="Summary video montage not found.")
    return FileResponse(path=str(mp4_path), media_type="video/mp4")
