"""
Media Synthesis Agent Node
--------------------------
Handles high-level video summarization, exportable Storyboard PDF creation,
and highlights video montage compilation with AI voiceover narration.
"""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from schemas.state_models import AgentState
from core.vector_db import VectorDBStore
from core.media_synthesis import MediaSynthesisEngine
from core.llm_factory import LLMFactory, extract_text_from_message_content

_vector_store = VectorDBStore()
_media_engine = MediaSynthesisEngine()


SUMMARY_PROMPT = """
You are a Video Summarization AI. Based on the chronological sequence of scenes below, generate:
1. Executive Summary: A concise 1-2 paragraph overview of the video narrative.
2. Key Highlights: 3 to 5 bullet points highlighting the most significant moments or insights.
3. Timeline: Key timestamps with a 1-line description of what happens.

Video Scenes:
{scenes_text}
"""


def media_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes summarization and media synthesis workflows:
    - Generates multi-part summary
    - Renders Storyboard PDF
    - Compiles summary video with narration if video_path is present
    """
    video_id = state.get("video_id") or "video"
    video_path = state.get("video_path")
    query = state.get("user_query", "")
    trace = list(state.get("reasoning_trace", []))
    artifacts = dict(state.get("artifacts", {}))

    trace.append(f"Fetching chronological scenes for video: {video_id}")
    scenes = _vector_store.get_all_scenes_for_video(video_id)

    if not scenes:
        if video_path:
            trace.append(f"No scene data found for '{video_id}'. Running automatic Tier 1 indexing...")
            from agents.ingestion_agent import ingestion_node
            ingest_res = ingestion_node(state)
            trace.extend(ingest_res.get("reasoning_trace", []))
            scenes = _vector_store.get_all_scenes_for_video(video_id)

    if not scenes:
        trace.append("No scene data found in vector database.")
        return {
            "final_answer": "No indexed scenes found for this video. Please ensure the video has been processed.",
            "reasoning_trace": trace
        }

    # Format scenes for LLM summarization
    scene_lines = []
    scene_dicts = []
    for s in scenes:
        meta = s.get("metadata", {})
        doc = s.get("document", "")
        idx = meta.get("scene_index", 0)
        st = meta.get("formatted_start", "00:00")
        et = meta.get("formatted_end", "00:00")
        kf = meta.get("keyframe_path")

        scene_lines.append(f"Scene {idx} [{st} - {et}]: {doc}")
        scene_dicts.append({
            "scene_index": idx,
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "formatted_start": st,
            "formatted_end": et,
            "description": doc,
            "keyframe_path": kf
        })

    trace.append("Generating executive summary and key highlights with LLM...")
    llm = LLMFactory.get_chat_model(temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content="You are a professional video analyst and editor."),
        HumanMessage(content=SUMMARY_PROMPT.format(scenes_text="\n".join(scene_lines[:40])))
    ])

    summary_text = extract_text_from_message_content(resp.content)

    # 1. Render Storyboard PDF
    trace.append("Rendering Storyboard PDF with scene keyframes...")
    pdf_path = _media_engine.create_storyboard_pdf(
        video_id=video_id,
        summary_text=summary_text[:1500],
        scenes=scene_dicts
    )
    if pdf_path:
        artifacts["storyboard_pdf"] = pdf_path
        trace.append(f"Storyboard PDF created: {pdf_path}")

    # 2. Compile Summary Video Montage if video source is available
    if video_path and any(k in query.lower() for k in ["video", "montage", "clip"]):
        trace.append("Compiling summary video montage with MoviePy and AI voiceover...")
        vid_summary_path = _media_engine.compile_summary_video(
            video_id=video_id,
            video_path=video_path,
            scene_segments=scene_dicts[:5],
            narration_text=summary_text[:500]
        )
        if vid_summary_path:
            artifacts["summary_video"] = vid_summary_path
            trace.append(f"Summary video montage rendered: {vid_summary_path}")

    return {
        "final_answer": summary_text,
        "artifacts": artifacts,
        "reasoning_trace": trace
    }
