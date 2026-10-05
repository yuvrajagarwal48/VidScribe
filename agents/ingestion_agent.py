"""
Progressive Ingestion Agent Node
--------------------------------
Implements Tier 1 rapid video indexing (scene boundaries, keyframes, and Whisper audio)
so that videos become queryable within seconds rather than forcing long upfront waits.
Tier 2 visual OCR and object graphs are cached lazily when needed.
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from threading import Thread

from schemas.state_models import AgentState
from schemas.video_models import SceneAnalysisResult
from core.video_engine import VideoEngine
from core.audio_engine import AudioEngine
from core.vision_engine import VisionEngine
from core.vector_db import VectorDBStore


from concurrent.futures import ThreadPoolExecutor, as_completed


def enrich_video_scene_graphs(video_id: str, scenes: list, keyframes: dict):
    """
    Background worker: Progressively runs Gemini Vision across scene keyframes in parallel
    to extract OCR text, visual descriptions, and scene graphs (objects with spatial relations),
    updating ChromaDB so future semantic searches match visual text & relationships.
    """
    print(f"Background worker: Starting visual scene graph enrichment for '{video_id}'...")
    vision_engine = VisionEngine()
    vector_store = VectorDBStore()

    def process_single_scene(scene):
        kf = keyframes.get(scene.scene_index)
        if not kf or not kf.image_path or not os.path.exists(kf.image_path):
            return
        try:
            analysis = vision_engine.inspect_frame_visuals(kf.image_path)
            vector_store.enrich_scene_visuals(
                video_id=video_id,
                scene_idx=scene.scene_index,
                scene_description=analysis.get("scene_description", ""),
                ocr_text=analysis.get("ocr_text", ""),
                objects=analysis.get("objects", []),
                keyframe_path=kf.image_path,
                formatted_start=scene.formatted_start,
                formatted_end=scene.formatted_end
            )
        except Exception as e:
            print(f"Notice: Scene graph enrichment note for scene {scene.scene_index}: {e}")

    # Process keyframes in parallel with 4 worker threads for high throughput
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_scene, s) for s in scenes]
        for f in as_completed(futures):
            pass

    print(f"Background worker: Completed visual scene graph enrichment for '{video_id}'.")


def ingestion_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes rapid Tier 1 indexing on the target video:
    1. Detects scenes and extracts middle keyframes.
    2. Runs Whisper to extract dialogue with timestamps.
    3. Populates ChromaDB so the video can be queried immediately.
    4. Triggers background worker for visual scene graph and OCR enrichment.
    """
    video_path = state.get("video_path")
    video_id = state.get("video_id") or (Path(video_path).stem if video_path else "unknown")
    trace = list(state.get("reasoning_trace", []))

    if not video_path:
        trace.append("Ingestion skipped: No video_path specified.")
        return {
            "final_answer": "No video file provided to process.",
            "reasoning_trace": trace
        }

    trace.append(f"Starting Tier 1 Rapid Ingestion for video: {video_id}")
    
    # 1. Video scenes and keyframes
    video_engine = VideoEngine(video_path)
    scenes = video_engine.detect_scenes()
    keyframes = video_engine.extract_keyframes()
    trace.append(f"Detected {len(scenes)} scenes and extracted keyframes.")

    # 2. Fast single-pass audio transcription (processes entire video in 3-5s)
    audio_engine = AudioEngine()
    vector_store = VectorDBStore()
    
    trace.append("Transcribing audio track via Whisper single-pass...")
    try:
        all_transcripts = audio_engine.transcribe_full_video(video_path)
        trace.append(f"Extracted {len(all_transcripts)} spoken dialogue segments.")
    except Exception as e:
        trace.append(f"Notice: Audio transcription note ({e}). Continuing with visual timeline.")
        all_transcripts = []

    indexed_count = 0
    for scene in scenes:
        kf = keyframes.get(scene.scene_index)
        kf_path = kf.image_path if kf else ""

        # Filter spoken segments belonging to this temporal scene
        scene_transcripts = [
            t for t in all_transcripts
            if (scene.start_time <= t.start_time <= scene.end_time)
            or (scene.start_time <= t.end_time <= scene.end_time)
            or (t.start_time <= scene.start_time and t.end_time >= scene.end_time)
        ]

        analysis = SceneAnalysisResult(
            video_id=video_id,
            scene_index=scene.scene_index,
            keyframe_path=kf_path,
            start_time=scene.start_time,
            end_time=scene.end_time,
            formatted_start=scene.formatted_start,
            formatted_end=scene.formatted_end,
            transcripts=scene_transcripts,
            ocr_texts=[],
            detected_objects=[]
        )

        doc_ids = vector_store.index_scene(analysis)
        if doc_ids:
            indexed_count += len(doc_ids)

    # 3. Launch background scene graph enrichment worker
    Thread(
        target=enrich_video_scene_graphs,
        args=(video_id, scenes, keyframes),
        daemon=True
    ).start()

    msg = f"Video '{video_id}' is ready! Indexed {len(scenes)} scenes in Tier 1. Background scene graph enrichment started."
    trace.append(msg)

    return {
        "final_answer": msg,
        "video_id": video_id,
        "reasoning_trace": trace
    }
