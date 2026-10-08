"""
VidScribe Evaluation & Benchmark Dashboard
-----------------------------------------
Interactive Streamlit application for multimodal video evaluation:
- Upload and inspect video files
- Ask questions and track real-time token consumption, phase latencies, cost, and throughput
- Visualize ChromaDB retrieved scenes, keyframe images, and cosine distance scores
- Run optional RAGAS evaluation (Faithfulness, Answer Relevance)
- Benchmark entire pipeline (Scene cuts, Whisper ASR, Vector Indexing, Summarization)

Launch:
    streamlit run eval_dashboard.py
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

# Prevent Unicode issues on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

import config
from core.vector_db import VectorDBStore
import types
try:
    import langchain_google_vertexai
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        _m = types.ModuleType("langchain_community.chat_models.vertexai")
        _m.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _m
except Exception:
    pass

from eval_tokens_and_rag import VidScribeEvaluator, TokenTrackerCallback, run_ragas_evaluation

# Page Configuration
st.set_page_config(
    page_title="VidScribe Evaluation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #38BDF8;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748B;
        font-size: 0.75rem;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Helper Functions
# =========================================================================

def get_available_videos():
    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    videos = []
    for f in config.DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_exts:
            videos.append(f)
    return videos


def get_video_stats(video_path: Path):
    try:
        engine = VideoEngine(str(video_path))
        return {
            "duration": engine.duration_seconds,
            "fps": engine.fps,
            "resolution": f"{engine.width}x{engine.height}",
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
        }
    except Exception:
        return {
            "duration": 0,
            "fps": 30,
            "resolution": "Unknown",
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
        }


# =========================================================================
# Sidebar: Video Management & Upload
# =========================================================================

st.sidebar.title("⚡ VidScribe Evaluator")
st.sidebar.caption("Multimodal Token, Latency & RAG Evaluation Harness")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Video Selection")

available_videos = get_available_videos()
video_names = [v.name for v in available_videos]

selected_video_name = None
if video_names:
    selected_video_name = st.sidebar.selectbox("Choose a Video to Evaluate:", video_names)

# File Uploader
st.sidebar.markdown("---")
st.sidebar.subheader("Upload New Video")
uploaded_file = st.sidebar.file_uploader("Upload MP4 / MOV for Evaluation", type=["mp4", "mov", "mkv", "webm"])

if uploaded_file is not None:
    dest_path = config.DATA_DIR / uploaded_file.name
    if not dest_path.exists():
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        st.rerun()

# Selected Video Status
selected_video_path = None
selected_video_id = None
if selected_video_name:
    selected_video_path = config.DATA_DIR / selected_video_name
    selected_video_id = selected_video_path.stem

    vstore = VectorDBStore()
    scenes = vstore.get_all_scenes_for_video(selected_video_id)
    is_indexed = len(scenes) > 0

    stats = get_video_stats(selected_video_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Video Metadata")
    st.sidebar.write(f"**Video ID:** `{selected_video_id}`")
    st.sidebar.write(f"**File Size:** {stats['size_mb']} MB")
    st.sidebar.write(f"**Duration:** {stats['duration']:.1f}s")
    st.sidebar.write(f"**Resolution:** {stats['resolution']}")
    st.sidebar.write(f"**Indexed Scenes:** {len(scenes)}")

    if not is_indexed:
        st.sidebar.warning("⚠️ Video is not indexed in ChromaDB.")
        if st.sidebar.button("⚡ Run Tier 1 Ingestion Now", type="primary"):
            with st.spinner("Running Tier 1 scene detection, Whisper ASR, and indexing..."):
                evaluator = VidScribeEvaluator(video_id=selected_video_id)
                res = evaluator.benchmark_ingestion_tier1()
                st.sidebar.success(f"Indexed {res.get('scenes_count', 0)} scenes in {res.get('total_tier1_seconds', 0)}s!")
                st.rerun()
    else:
        st.sidebar.success(f"✅ Indexed ({len(scenes)} scenes in ChromaDB)")


# =========================================================================
# Main Dashboard
# =========================================================================

st.title("🔬 VidScribe Evaluation & Benchmarking Dashboard")

if not selected_video_path:
    st.info("👈 Please select or upload a video in the sidebar to begin evaluation.")
    st.stop()

tab_qa, tab_pipeline, tab_reports = st.tabs([
    "💬 Interactive QA & Token Evaluation",
    "⚡ Full Pipeline Benchmark",
    "📊 Audit Reports & History"
])


# =========================================================================
# TAB 1: Interactive QA & Token Evaluation
# =========================================================================

with tab_qa:
    st.subheader(f"Query Evaluation for: `{selected_video_id}`")

    # Sample query selector or custom input
    sample_queries = [
        "Custom query...",
        "Who won the match and what was the final score?",
        "At what timestamp did the key event take place?",
        "What visual details or graphics are visible on screen?",
        "Summarize the main turning points of this video."
    ]
    query_choice = st.selectbox("Quick Sample Query or Custom:", sample_queries)

    if query_choice == "Custom query...":
        user_query = st.text_input("Enter your evaluation question:", value="Who won the match and what was the score?")
    else:
        user_query = st.text_input("Evaluation question:", value=query_choice)

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        ground_truth = st.text_input("Optional Expected Ground Truth (for reference / RAGAS comparison):", "")
    with col_q2:
        run_ragas = st.checkbox("Include Ragas Evaluation (Faithfulness & Relevance)", value=False)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        start_eval = st.button("🚀 Evaluate Query", type="primary", use_container_width=True)

    if start_eval and user_query.strip():
        evaluator = VidScribeEvaluator(video_id=selected_video_id)

        with st.spinner("Executing Multimodal RAG with token & latency tracking..."):
            query_result = evaluator.benchmark_query_rag(query=user_query)

        # -------------------------------------------------------------
        # 1. Performance & Cost Cards
        # -------------------------------------------------------------
        st.markdown("### 📊 Real-Time Metrics & Cost Breakdown")

        # Sum metrics from tracker
        total_inp = 0
        total_out = 0
        total_tok = 0
        total_cost = 0.0
        total_time = query_result.get("e2e_query_latency_s", 0.0)

        for p in evaluator.tracker.phases.values():
            total_inp += p.input_tokens
            total_out += p.output_tokens
            total_tok += p.total_tokens
            total_cost += p.cost_usd

        tps = (total_out / total_time) if total_time > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Latency", f"{total_time:.2f} s", f"{query_result.get('vector_search_latency_ms', 0):.1f} ms search")
        with c2:
            st.metric("Input Tokens", f"{total_inp:,}", "Prompt + Keyframes")
        with c3:
            st.metric("Output Tokens", f"{total_out:,}", f"{tps:.1f} tokens/s")
        with c4:
            st.metric("Total Tokens", f"{total_tok:,}", f"{len(evaluator.tracker.phases)} phase(s)")
        with c5:
            st.metric("Estimated Cost", f"${total_cost:.5f}", "Gemini Flash rates")

        # -------------------------------------------------------------
        # 2. Generated Grounded Answer & Timestamp Citations
        # -------------------------------------------------------------
        st.markdown("### 💡 Grounded Response")
        st.markdown(f"> {query_result.get('answer', '')}")

        c_cite1, c_cite2 = st.columns(2)
        with c_cite1:
            st.write(f"**Grounded Citations:** {query_result.get('citations_count', 0)} scene(s)")
        with c_cite2:
            has_ts = query_result.get('has_timestamp_citation', False)
            if has_ts:
                st.success("✅ Valid timestamp citations formatted in response")
            else:
                st.warning("⚠️ No explicit `[MM:SS]` timestamp found in answer")

        # -------------------------------------------------------------
        # 3. Retrieved Candidate Scenes & Keyframe Previews
        # -------------------------------------------------------------
        st.markdown("### 🔍 Retrieved Evidence & Keyframe Cards")
        segs = query_result.get("grounded_segments", [])
        if segs:
            cols = st.columns(min(len(segs), 4))
            for i, seg in enumerate(segs[:4]):
                with cols[i]:
                    img_path = seg.get("source_preview")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Scene {seg.get('scene_index', i) + 1} [{seg.get('formatted_start')} - {seg.get('formatted_end')}]")
                    else:
                        st.info(f"Scene {seg.get('scene_index', i) + 1}\n[{seg.get('formatted_start')} - {seg.get('formatted_end')}]")
        else:
            st.info("No scene citations available.")

        # -------------------------------------------------------------
        # 4. Phase Breakdown Table
        # -------------------------------------------------------------
        st.markdown("### 📋 Granular Phase Breakdown")
        phase_rows = []
        for name, p in evaluator.tracker.phases.items():
            phase_rows.append({
                "Phase / Task": name,
                "Calls": p.call_count,
                "Input Tokens": p.input_tokens,
                "Output Tokens": p.output_tokens,
                "Total Tokens": p.total_tokens,
                "Cost ($)": f"${p.cost_usd:.5f}",
                "Latency (s)": f"{p.latency_seconds:.2f}s",
                "Throughput (TPS)": f"{p.tokens_per_sec:.1f}" if p.output_tokens > 0 else "-"
            })
        st.table(phase_rows)

        # -------------------------------------------------------------
        # 5. Optional Ragas Evaluation
        # -------------------------------------------------------------
        if run_ragas:
            st.markdown("### 🏆 Ragas Evaluation Assessment")
            with st.spinner("Computing Ragas evaluation metrics with Gemini Evaluator..."):
                contexts = query_result.get("retrieved_contexts", [])
                if not contexts:
                    contexts = [s.get("source_preview", "") or f"Scene {s.get('scene_index')}" for s in segs]

                ragas_res = run_ragas_evaluation(
                    query=user_query,
                    response=query_result.get("answer", ""),
                    retrieved_contexts=contexts,
                    ground_truth=ground_truth if ground_truth else None
                )

                if ragas_res:
                    faith_score = ragas_res.get("faithfulness")
                    if faith_score is not None and not str(faith_score).lower() == "nan":
                        try:
                            score_val = float(faith_score)
                            r_c1, r_c2 = st.columns([1, 2])
                            with r_c1:
                                st.metric("Ragas Faithfulness", f"{score_val * 100:.1f}%", "100% = No Hallucination")
                            with r_c2:
                                st.progress(min(1.0, max(0.0, score_val)), text=f"Context Grounding: {score_val * 100:.1f}%")
                        except Exception:
                            pass
                    st.json(ragas_res)
                else:
                    st.warning("Ragas evaluation returned no score. Check logs for details.")

        # Auto export report
        exported = evaluator.export_reports(query_info=query_result)
        st.success(f"Report saved to `{exported['markdown_path']}`")


# =========================================================================
# TAB 2: Full Pipeline Benchmark
# =========================================================================

with tab_pipeline:
    st.subheader(f"Full End-to-End Pipeline Latency & Token Benchmark: `{selected_video_id}`")
    st.markdown("""
    This benchmarks every stage of the pipeline consecutively:
    1. **PySceneDetect & Keyframe Midpoints** (CV extraction latency)
    2. **Whisper Speech Recognition** (Audio ASR latency + Real-Time Factor RTF)
    3. **ChromaDB Vector Indexing** (Embedding & storage time)
    4. **Tier 2 Gemini Vision** (Multimodal image inspection tokens & latency)
    5. **Multimodal RAG QA** (Retrieval, prompt tokens, completion tokens, TPS)
    6. **Full Video Narrative Summarization** (Chronological scene synthesis)
    """)

    if st.button("⚡ Run Entire Pipeline Benchmark", type="primary"):
        evaluator = VidScribeEvaluator(video_id=selected_video_id)

        progress_bar = st.progress(0, text="Starting Full Pipeline Benchmark...")

        # 1. Ingestion Tier 1
        progress_bar.progress(20, text="1/4 Benchmarking Scene Detection & Whisper ASR...")
        ingest_res = evaluator.benchmark_ingestion_tier1()

        # 2. Tier 2 Vision
        progress_bar.progress(40, text="2/4 Benchmarking Gemini Vision Keyframe Inspection...")
        vis_res = evaluator.benchmark_vision_enrichment(sample_frames=2)

        # 3. Query QA
        progress_bar.progress(65, text="3/4 Benchmarking Semantic Retrieval & Multimodal RAG...")
        q_res = evaluator.benchmark_query_rag(query="Who won the match and what was the score?")

        # 4. Summarization
        progress_bar.progress(85, text="4/4 Benchmarking Full Video Summarization...")
        sum_res = evaluator.benchmark_summarization()

        progress_bar.progress(100, text="Pipeline Benchmark Complete!")

        # Display High-Level Summary Cards
        st.markdown("### 🏆 Full Pipeline Benchmark Results")

        grand_calls = sum(p.call_count for p in evaluator.tracker.phases.values())
        grand_in = sum(p.input_tokens for p in evaluator.tracker.phases.values())
        grand_out = sum(p.output_tokens for p in evaluator.tracker.phases.values())
        grand_cost = sum(p.cost_usd for p in evaluator.tracker.phases.values())
        grand_time = sum(p.latency_seconds for p in evaluator.tracker.phases.values())
        grand_tps = (grand_out / grand_time) if grand_time > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Pipeline Time", f"{grand_time:.2f} s", f"{len(evaluator.tracker.phases)} phases evaluated")
        with c2:
            st.metric("Total Tokens", f"{(grand_in + grand_out):,}", f"{grand_in:,} in / {grand_out:,} out")
        with c3:
            st.metric("Total Cost", f"${grand_cost:.5f}", "USD")
        with c4:
            st.metric("Audio RTF Factor", f"{ingest_res.get('real_time_factor_rtf', 0):.1f}x", "Faster than real-time playback")

        # Full Phase Table
        st.markdown("### 📊 Comprehensive Phase Latency & Token Table")
        phase_data = []
        for name, p in evaluator.tracker.phases.items():
            phase_data.append({
                "Pipeline Phase": name,
                "Calls": p.call_count,
                "Input Tokens": f"{p.input_tokens:,}",
                "Output Tokens": f"{p.output_tokens:,}",
                "Total Tokens": f"{p.total_tokens:,}",
                "Latency": f"{p.latency_seconds:.2f}s",
                "Cost": f"${p.cost_usd:.5f}",
                "TPS": f"{p.tokens_per_sec:.1f}" if p.output_tokens > 0 else "-"
            })
        st.table(phase_data)

        # Export report
        exported = evaluator.export_reports(query_info=q_res)
        st.success(f"Audit report exported to: `{exported['markdown_path']}`")


# =========================================================================
# TAB 3: Historical Audit Reports
# =========================================================================

with tab_reports:
    st.subheader("📑 Saved Audit Reports & Performance Logs")

    reports_dir = config.BASE_DIR / "reports"
    if reports_dir.exists():
        md_files = sorted(list(reports_dir.glob("*.md")), key=lambda p: p.stat().st_mtime, reverse=True)
        if md_files:
            selected_md = st.selectbox("Select Report to View:", [f.name for f in md_files])
            report_file_path = reports_dir / selected_md

            with open(report_file_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            st.download_button(
                label="📥 Download Markdown Report",
                data=report_content,
                file_name=selected_md,
                mime="text/markdown"
            )

            st.markdown("---")
            st.markdown(report_content)
        else:
            st.info("No audit reports found in `reports/` yet. Run a query evaluation to generate one.")
    else:
        st.info("Reports directory not created yet.")
