"""
FastAPI Application Entrypoint
------------------------------
Configures FastAPI, CORS middleware, REST API routes, WebSocket streaming,
and serves the modern Web UI static assets.
"""

import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config
from backend.routes import router
from agents.workflow import videorag_agent_app
from schemas.rag_models import GroundedSegment
from core.llm_factory import extract_text_from_message_content

from backend.session_memory import (
    get_session_messages,
    append_session_interaction,
    clear_session_memory,
    clear_video_sessions
)

# Initialize FastAPI application
app = FastAPI(
    title="VidScribe VideoRAG API",
    description="VidScribe: Agentic Multimodal Video Retrieval-Augmented Generation & Synthesis",
    version="2.0.0"
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST endpoints under /api
app.include_router(router)


# =========================================================================
# WebSocket Channel for Streaming Chat & Live Agent Thought Traces
# =========================================================================

@app.websocket("/ws/chat/{video_id}")
async def websocket_chat_endpoint(websocket: WebSocket, video_id: str):
    """
    Real-time streaming WebSocket endpoint.
    Streams agent thought logs, intermediate tool traces, and grounded tokens.
    Maintains conversational memory per session across multi-turn interactions.
    """
    await websocket.accept()

    try:
        while True:
            # Receive user query and optional session_id from frontend
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            user_query = payload.get("query", "").strip()
            session_id = payload.get("session_id") or f"session_{video_id}"

            if not user_query:
                continue

            # Emit initial status
            await websocket.send_json({
                "type": "trace",
                "message": f"[SUPERVISOR] 🎯 Orchestrator analyzing query: '{user_query}'"
            })

            # Locate video file if video_id is provided
            video_path = None
            if video_id and video_id != "all":
                for f in config.DATA_DIR.iterdir():
                    if f.stem == video_id:
                        video_path = str(f)
                        break

            # Implicit Auto-Indexing: If video is not yet indexed, index it automatically on-the-fly
            if video_path and video_id != "all":
                from core.vector_db import VectorDBStore
                vstore = VectorDBStore()
                existing_scenes = vstore.get_all_scenes_for_video(video_id)
                if not existing_scenes:
                    await websocket.send_json({
                        "type": "trace",
                        "message": f"[INGESTION] ⚡ Video '{video_id}' is unindexed. Automatically performing Tier 1 indexing..."
                    })
                    from agents.ingestion_agent import ingestion_node
                    ingest_state = {
                        "video_path": video_path,
                        "video_id": video_id,
                        "reasoning_trace": []
                    }
                    ingest_res = ingestion_node(ingest_state)
                    for tr in ingest_res.get("reasoning_trace", []):
                        await websocket.send_json({
                            "type": "trace",
                            "message": f"[INGESTION] {tr}"
                        })

            # Retrieve prior conversation turns from session memory
            prior_messages = get_session_messages(session_id)

            # Prepare state
            init_state = {
                "messages": prior_messages,
                "video_id": None if video_id == "all" else video_id,
                "video_path": video_path,
                "user_query": user_query,
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

            config_dict = {"configurable": {"thread_id": f"ws_{session_id}"}}

            # Stream agent node transitions
            final_res = None
            for event in videorag_agent_app.stream(init_state, config_dict):
                for node_name, node_output in event.items():
                    trace_items = node_output.get("reasoning_trace", [])
                    if trace_items:
                        latest_trace = trace_items[-1]
                        await websocket.send_json({
                            "type": "trace",
                            "message": latest_trace
                        })
                    final_res = node_output

            # Emit final grounded response
            if final_res and final_res.get("final_answer"):
                answer = extract_text_from_message_content(final_res["final_answer"])
                segments = final_res.get("grounded_segments", [])

                # Save turn to conversational session memory
                append_session_interaction(session_id, user_query, answer)

                # Stream response text chunks smoothly
                chunk_size = 30
                for i in range(0, len(answer), chunk_size):
                    await websocket.send_json({
                        "type": "token",
                        "token": answer[i:i + chunk_size]
                    })
                    await asyncio.sleep(0.015)

                # Stream grounded citation segment pills
                await websocket.send_json({
                    "type": "citations",
                    "segments": segments
                })

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        try:
            print(f"WebSocket client disconnected for video: {video_id}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"WebSocket error: {e}")
        except Exception:
            pass
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# =========================================================================
# Serve Modern Web UI Static Files
# =========================================================================

# Ensure frontend directory exists
if config.FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        """Serves the main single-page web UI with no-cache headers."""
        index_file = config.FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(
                str(index_file),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        return {"message": "VidScribe VideoRAG API is running. Place index.html in frontend/"}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting VidScribe Web Application at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run("backend.app:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=True)
