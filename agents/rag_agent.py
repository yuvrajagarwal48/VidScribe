"""
Multimodal RAG Agent Node
-------------------------
Implements Corrective / Self-Reflective RAG (CRAG):
1. Performs semantic search across multimodal scene vectors in ChromaDB.
2. Identifies if visual inspection or OCR is required for the user's query.
3. Dynamically inspects candidate keyframes on-the-fly.
4. Synthesizes a grounded answer citing precise timestamps and scene boundaries.
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from schemas.state_models import AgentState
from core.vector_db import VectorDBStore
from core.vision_engine import VisionEngine
from core.llm_factory import LLMFactory, extract_text_from_message_content

_vector_store = VectorDBStore()
_vision_engine = VisionEngine()


RAG_SYSTEM_PROMPT = """
You are VidScribe AI Analyst, an intelligent Multimodal Video Intelligence Assistant.
Your mission is to answer user questions about video content accurately, concisely, and with grounded evidence from BOTH the provided scene transcripts and the attached visual keyframes.

Guidelines:
1. Ground your answers in both the spoken dialogue context AND the visual keyframes (inspecting on-screen text, scoreboards, player names, jerseys, score numbers, banners, slide deck text, objects, and visual actions).
2. Whenever you mention an event, action, score, or dialogue, ALWAYS reference its exact timestamp (e.g., [02:15] or [01:38 - 01:45]).
3. If visual details (like player names or scores on a scoreboard) are visible in the attached images, state them explicitly to directly answer questions about who, what, and how much.
4. Maintain conversational context across multi-turn exchanges. If the user asks a follow-up question, resolve pronouns and references seamlessly using the conversation history.
5. Respond in a direct, helpful, and grounded tone.
"""


def reformulate_query_with_history(query: str, history_text: str) -> str:
    """
    If the user's question references past turns (pronouns, ellipses, follow-ups),
    reformulate it into a self-contained search query for vector retrieval.
    """
    if not history_text.strip():
        return query

    query_lower = query.lower().strip()
    words = query_lower.split()
    pronouns = {"he", "she", "it", "they", "them", "his", "her", "their", "that", "this", "these", "those", "him"}
    followup_cues = ["who was", "what about", "and then", "after that", "what did", "which one", "why did", "how many", "tell me more", "explain that", "show that", "what else"]
    
    needs_rewrite = (
        bool(set(words) & pronouns) or
        any(cue in query_lower for cue in followup_cues) or
        len(words) <= 4
    )

    if not needs_rewrite:
        return query

    try:
        fast_llm = LLMFactory.get_chat_model(temperature=0.0)
        rewrite_prompt = (
            "You are a search query reformulation specialist for a Video Intelligence retrieval engine.\n"
            "Given the recent conversation history and a follow-up question, rewrite the follow-up question into a standalone, concise keyword search query.\n"
            "Preserve key names, entities, objects, or actions from the previous turns so the query can retrieve relevant video scenes.\n"
            "If the question is already clear and self-contained, output it unchanged.\n"
            "Output ONLY the reformulated query text, nothing else.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Follow-up Question: {query}\n"
            "Standalone Search Query:"
        )
        res = fast_llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = extract_text_from_message_content(res.content).strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 2:
            return rewritten
    except Exception:
        pass

    return query


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG node: queries ChromaDB, performs dynamic keyframe inspection if visual details
    are requested, and generates an answer with grounded timestamp citations.
    Supports short-term conversational memory across turns.
    """
    query = state.get("user_query", "")
    video_id = state.get("video_id")
    trace = list(state.get("reasoning_trace", []))

    # 1. Parse prior conversation dialogue history from state["messages"]
    raw_messages = state.get("messages", [])
    history_turns: List[str] = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            history_turns.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_turns.append(f"VidScribe: {extract_text_from_message_content(msg.content)}")
        elif isinstance(msg, dict):
            role = msg.get("role", "User")
            content = msg.get("content", "")
            history_turns.append(f"{role.capitalize()}: {content}")

    # Retain the last 6 turns for conversational context
    recent_history = history_turns[-6:] if history_turns else []
    history_text = "\n".join(recent_history)

    # 2. Contextual Query Reformulation for Vector Search if follow-up detected
    search_query = query
    if recent_history:
        search_query = reformulate_query_with_history(query, history_text)
        if search_query != query:
            trace.append(f"[MEMORY] 🧠 Follow-up detected ({len(recent_history)} prior turns). Reformulated search: '{search_query}'")
        else:
            trace.append(f"[MEMORY] 🧠 Dialogue memory active ({len(recent_history)} prior turns). Query evaluated as self-contained.")

    trace.append(f"[RAG] 🔍 Searching ChromaDB multimodal vector store for: '{search_query}'")
    hits = _vector_store.query(query_text=search_query, video_id=video_id, top_k=4)
    
    if not hits:
        video_path = state.get("video_path")
        if video_path:
            trace.append(f"[INGESTION] ⚡ No indexed scenes found for '{video_id}'. Running automatic Tier 1 indexing...")
            from agents.ingestion_agent import ingestion_node
            ingest_res = ingestion_node(state)
            trace.extend(ingest_res.get("reasoning_trace", []))
            hits = _vector_store.query(query_text=search_query, video_id=video_id, top_k=4)

    if not hits:
        trace.append("[RAG] ⚠️ No matching scenes found in vector store.")
        no_hit_msg = "I couldn't find any scenes in the video matching your question. Please ensure the video has been uploaded and processed."
        return {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=no_hit_msg)
            ],
            "final_answer": no_hit_msg,
            "grounded_segments": [],
            "reasoning_trace": trace
        }

    # Temporal Outcome Anchoring: Ensure concluding scenes are in context for outcome/score queries
    query_lower = search_query.lower()
    is_outcome_query = any(k in query_lower for k in [
        "how many overs", "overs did", "overs taken", "balls left", "balls remaining", 
        "who won", "winner", "result", "how did it end", "final score", "conclude", 
        "finish", "winning run", "winning shot", "end of", "chase", "chased"
    ])

    if is_outcome_query and video_id:
        concluding_scenes = _vector_store.get_concluding_scenes_for_video(video_id, count=2)
        existing_indices = {h.get("metadata", {}).get("scene_index") for h in hits}
        anchored_count = 0
        for s in concluding_scenes:
            s_idx = s.get("metadata", {}).get("scene_index")
            if s_idx not in existing_indices:
                hits.append(s)
                existing_indices.add(s_idx)
                anchored_count += 1
        if anchored_count > 0:
            trace.append(f"[RAG] ⏱️ Query inquires about outcome. Temporally anchored {anchored_count} concluding scene(s).")

    context_blocks = []
    grounded_segments = []
    message_parts: List[Dict[str, Any]] = []

    # 3. Compile chronological scene context blocks
    for idx, hit in enumerate(hits):
        doc = hit.get("document", "")
        meta = hit.get("metadata", {})
        
        scene_idx = meta.get("scene_index", idx)
        st_fmt = meta.get("formatted_start", "00:00")
        et_fmt = meta.get("formatted_end", "00:00")
        kf_path = meta.get("keyframe_path")

        scene_text = f"Scene {scene_idx + 1} [{st_fmt} - {et_fmt}]: {doc}"
        context_blocks.append(scene_text)

        grounded_segments.append({
            "video_id": meta.get("video_id", video_id or "video"),
            "scene_index": scene_idx,
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "formatted_start": st_fmt,
            "formatted_end": et_fmt,
            "modality_type": meta.get("type", "combined"),
            "source_preview": kf_path
        })

    trace.append(f"[RAG] 📊 Synthesizing grounded context across {len(context_blocks)} retrieved scenes.")
    
    context_str = "\n\n".join(context_blocks)
    
    # 4. Build multimodal payload with dialogue history + scene transcripts + candidate keyframes
    history_section = f"Recent Conversation History:\n{history_text}\n\n" if history_text else ""

    intro_text = f"""{history_section}Provided Video Dialogue & Scene Transcripts:
{context_str}

User Question: {query}

Instructions:
Synthesize an accurate, grounded answer using the dialogue transcripts, conversation history, and attached visual keyframes below.
- If this is a follow-up question, seamlessly resolve pronouns/references using the conversation history.
- Carefully inspect on-screen graphics, scoreboards, player names, jersey names, runs/scores, statistics, banners, slide deck text, objects, and visual actions.
- For cricket or sports match outcome queries:
  Examine the on-screen scoreboard overlay at the end of the chase and state the exact numbers and timestamps.
- If an entity or answer is shown visually (e.g. scoreboard name, runs, or overs), explicitly state it.
- ALWAYS cite the exact timestamp (e.g., [01:38 - 01:45] or [04:12 - 04:31]).
"""
    message_parts.append({"type": "text", "text": intro_text})

    # Attach keyframe images for top candidate scenes + concluding scenes
    attached_count = 0
    attached_scene_indices = set()
    for hit in hits:
        meta = hit.get("metadata", {})
        s_idx = meta.get("scene_index", 0)
        kf_path = meta.get("keyframe_path")
        if s_idx in attached_scene_indices or attached_count >= 4:
            continue
        if kf_path and os.path.exists(kf_path):
            b64_img = VisionEngine.encode_image_to_base64(kf_path)
            if b64_img:
                st_fmt = meta.get("formatted_start", "00:00")
                et_fmt = meta.get("formatted_end", "00:00")
                message_parts.append({
                    "type": "text",
                    "text": f"--- Visual Frame for Scene {s_idx + 1} [{st_fmt} - {et_fmt}] ---"
                })
                message_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
                attached_count += 1
                attached_scene_indices.add(s_idx)

    if attached_count > 0:
        trace.append(f"[VISION] 👁️ Inspected {attached_count} candidate keyframe image(s) for visual details.")

    llm = LLMFactory.get_chat_model(temperature=0.2)
    trace.append("[ANALYST] 🧠 Gemini Multimodal Analyst generating grounded response...")

    resp = llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=message_parts)
    ])

    final_text = extract_text_from_message_content(resp.content)
    trace.append("[ANALYST] ✅ Grounded answer formulated with precise timestamps.")

    return {
        "messages": [
            HumanMessage(content=query),
            AIMessage(content=final_text)
        ],
        "final_answer": final_text,
        "grounded_segments": grounded_segments,
        "reasoning_trace": trace
    }
