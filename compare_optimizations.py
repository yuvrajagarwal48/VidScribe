"""
VidScribe Performance & Optimization Comparison Harness
--------------------------------------------------------
Loads a baseline evaluation report (from previous runs or eval_dashboard)
and runs the same benchmarks against the newly optimized codebase:
- Measures ChromaDB batch vector upsert latency (before vs after)
- Measures Whisper ASR speech-to-text latency (before vs after)
- Measures Multimodal Gemini RAG query latency & tokens (before vs after)
- Computes percentage speedup, latency delta, and token/cost differentials
- Exports comparison tables to Terminal, Markdown, and JSON.

Usage:
    python compare_optimizations.py
    python compare_optimizations.py --baseline reports/latest_metrics.json
    python compare_optimizations.py --video "sample_videos/demo.mp4" --question "Who won the match?"
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# UTF-8 terminal support on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from eval_tokens_and_rag import VidScribeEvaluator

REPORTS_DIR = config.BASE_DIR / "reports"
DEFAULT_BASELINE_PATH = REPORTS_DIR / "latest_metrics.json"


def load_baseline_report(baseline_path: Path) -> Dict[str, Any]:
    """Loads previous benchmark report JSON."""
    if not baseline_path.exists():
        # Fallback to any metrics_*.json in reports directory
        json_candidates = list(REPORTS_DIR.glob("metrics_*.json"))
        if json_candidates:
            baseline_path = json_candidates[0]
            print(f"[*] Default baseline not found. Using candidate: {baseline_path.name}")
        else:
            raise FileNotFoundError(
                f"No baseline JSON found at {baseline_path} or in {REPORTS_DIR}. "
                "Please run an initial benchmark via eval_tokens_and_rag.py or eval_dashboard.py first."
            )

    print(f"[*] Loaded Baseline Benchmark: {baseline_path.name}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_delta_pct(old_val: float, new_val: float, lower_is_better: bool = True) -> str:
    """Calculates percentage change with visual indicator."""
    if old_val == 0:
        return "N/A"
    diff = new_val - old_val
    pct = (diff / old_val) * 100.0

    if lower_is_better:
        if pct < -0.1:
            return f"🟢 {abs(pct):.1f}% faster (-{abs(diff):.2f}s)"
        elif pct > 0.1:
            return f"🔴 {abs(pct):.1f}% slower (+{diff:.2f}s)"
        else:
            return f"⚪ identical"
    else:
        if pct > 0.1:
            return f"🟢 +{pct:.1f}%"
        elif pct < -0.1:
            return f"🔴 {pct:.1f}%"
        else:
            return f"⚪ identical"


def compare_benchmarks(
    baseline_data: Dict[str, Any],
    video_id: Optional[str] = None,
    question: Optional[str] = None,
    skip_ingestion: bool = False
) -> Dict[str, Any]:
    """
    Runs the current optimized pipeline and compares against the baseline.
    """
    target_video_id = video_id or baseline_data.get("video_id") or "Cricket World Cup 2011 India vs Aus Highlights ｜ Thrilling Match"
    target_query = (
        question or
        baseline_data.get("query_evaluation", {}).get("query") or
        "Who won the match and what was the score?"
    )

    print("\n" + "=" * 70)
    print("        🚀 Running VidScribe Optimization Comparison Harness        ")
    print("=" * 70)
    print(f"[*] Target Video : {target_video_id}")
    print(f"[*] Test Question: {target_query}")
    print(f"[*] Baseline Date: {baseline_data.get('timestamp', 'Unknown')}")
    print("=" * 70)

    evaluator = VidScribeEvaluator(video_id=target_video_id)

    # 1. Benchmark Tier 1 Ingestion (Whisper + Batch ChromaDB)
    ingestion_results = {}
    if not skip_ingestion:
        print("\n[Phase 1/2] Benchmarking Tier 1 Ingestion (SceneDetect + Whisper + Batch Vector Upsert)...")
        ingestion_results = evaluator.benchmark_ingestion_tier1()
    else:
        print("\n[*] Skipping Tier 1 ingestion benchmark (--skip-ingestion specified).")

    # 2. Benchmark Query Multimodal RAG
    print("\n[Phase 2/2] Benchmarking Semantic Retrieval & Gemini Multimodal RAG...")
    query_result = evaluator.benchmark_query_rag(query=target_query)

    # 3. Build Comparison Structures
    baseline_phases = {p["phase"]: p for p in baseline_data.get("phases", [])}
    current_phases = evaluator.tracker.phases

    comparison_rows = []
    all_phase_names = list(baseline_phases.keys())
    for name in current_phases.keys():
        if name not in all_phase_names:
            all_phase_names.append(name)

    total_base_latency = 0.0
    total_opt_latency = 0.0

    for phase_name in all_phase_names:
        base_p = baseline_phases.get(phase_name, {})
        opt_p = current_phases.get(phase_name)

        base_lat = float(base_p.get("latency_s", 0.0))
        opt_lat = float(opt_p.latency_seconds) if opt_p else 0.0

        total_base_latency += base_lat
        total_opt_latency += opt_lat

        delta_str = format_delta_pct(base_lat, opt_lat, lower_is_better=True) if base_lat > 0 and opt_lat > 0 else "N/A"

        comparison_rows.append({
            "phase": phase_name,
            "baseline_latency_s": round(base_lat, 2),
            "optimized_latency_s": round(opt_lat, 2),
            "delta_display": delta_str,
            "baseline_tokens": int(base_p.get("total_tokens", 0)),
            "optimized_tokens": int(opt_p.total_tokens) if opt_p else 0,
            "baseline_cost": float(base_p.get("cost_usd", 0.0)),
            "optimized_cost": float(opt_p.cost_usd) if opt_p else 0.0
        })

    # Summary Totals
    base_summary = baseline_data.get("summary", {})
    base_total_tokens = base_summary.get("grand_total_tokens", 0)
    opt_total_tokens = sum(p.total_tokens for p in current_phases.values())

    base_total_cost = base_summary.get("total_cost_usd", 0.0)
    opt_total_cost = sum(p.cost_usd for p in current_phases.values())

    # Terminal Output Display
    print("\n" + "=" * 80)
    print("                 📈 OPTIMIZATION BENCHMARK COMPARISON               ")
    print("=" * 80)
    print(f"{'Pipeline Phase':<32} | {'Baseline (s)':<12} | {'Optimized (s)':<13} | {'Improvement'}")
    print("-" * 80)

    for r in comparison_rows:
        if r["baseline_latency_s"] > 0 or r["optimized_latency_s"] > 0:
            print(f"{r['phase']:<32} | {r['baseline_latency_s']:>10.2f}s | {r['optimized_latency_s']:>11.2f}s | {r['delta_display']}")

    print("-" * 80)
    time_saved = total_base_latency - total_opt_latency
    total_pct_saved = ((time_saved) / total_base_latency * 100) if total_base_latency > 0 else 0
    print(f"{'TOTAL PIPELINE EXECUTION':<32} | {total_base_latency:>10.2f}s | {total_opt_latency:>11.2f}s | 🟢 {total_pct_saved:.1f}% faster (-{time_saved:.2f}s)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("                 💰 TOKEN & COST CONSUMPTION COMPARISON             ")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Baseline':<12} | {'Optimized':<13} | {'Delta'}")
    print("-" * 80)
    print(f"{'Grand Total Tokens':<32} | {base_total_tokens:>12,d} | {opt_total_tokens:>13,d} | {opt_total_tokens - base_total_tokens:+d}")
    print(f"{'Estimated Cost (USD)':<32} | ${base_total_cost:>11.6f} | ${opt_total_cost:>12.6f} | ${opt_total_cost - base_total_cost:+.6f}")
    print("=" * 80)

    # Answers Comparison
    print("\n" + "=" * 80)
    print("                 🧠 QUERY ANSWER & CITATIONS COMPARISON             ")
    print("=" * 80)
    base_ans = baseline_data.get("query_evaluation", {}).get("answer", "N/A")
    opt_ans = query_result.get("answer", "N/A")
    print(f"[*] Baseline Answer :\n    {base_ans}\n")
    print(f"[*] Optimized Answer:\n    {opt_ans}\n")
    print(f"[*] Citations: Baseline = {baseline_data.get('query_evaluation', {}).get('citations_count', 0)} scenes | Optimized = {query_result.get('citations_count', 0)} scenes")
    print("=" * 80)

    # 4. Export Comparison Reports (Markdown & JSON)
    comparison_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_id": target_video_id,
        "query": target_query,
        "summary": {
            "baseline_total_latency_s": round(total_base_latency, 2),
            "optimized_total_latency_s": round(total_opt_latency, 2),
            "total_time_saved_s": round(time_saved, 2),
            "total_speedup_percent": round(total_pct_saved, 2),
            "baseline_total_tokens": base_total_tokens,
            "optimized_total_tokens": opt_total_tokens,
            "baseline_cost_usd": round(base_total_cost, 6),
            "optimized_cost_usd": round(opt_total_cost, 6)
        },
        "phase_comparisons": comparison_rows,
        "baseline_answer": base_ans,
        "optimized_answer": opt_ans
    }

    # Save JSON
    out_json = REPORTS_DIR / "optimization_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)

    # Save Markdown
    out_md = REPORTS_DIR / "optimization_comparison.md"
    md_content = f"""# 🚀 VidScribe Optimization Comparison Report

**Generated:** {comparison_payload['timestamp']}  
**Target Video:** `{target_video_id}`  
**Evaluation Question:** *"{target_query}"*  

---

## ⚡ Latency & Speedup Breakdown

| Pipeline Phase | Baseline Latency (s) | Optimized Latency (s) | Improvement |
| :--- | :---: | :---: | :---: |
"""
    for r in comparison_rows:
        if r["baseline_latency_s"] > 0 or r["optimized_latency_s"] > 0:
            md_content += f"| **{r['phase']}** | {r['baseline_latency_s']:.2f}s | {r['optimized_latency_s']:.2f}s | {r['delta_display']} |\n"

    md_content += f"""| **🏆 TOTAL PIPELINE** | **{total_base_latency:.2f}s** | **{total_opt_latency:.2f}s** | **{total_pct_saved:.1f}% faster (-{time_saved:.2f}s)** |

---

## 💰 Tokens & Estimated Cost

| Metric | Baseline | Optimized | Delta |
| :--- | :---: | :---: | :---: |
| **Grand Total Tokens** | {base_total_tokens:,} | {opt_total_tokens:,} | {opt_total_tokens - base_total_tokens:+d} |
| **Estimated Cost (USD)** | ${base_total_cost:.6f} | ${opt_total_cost:.6f} | ${opt_total_cost - base_total_cost:+.6f} |

---

## 🧠 Generated Responses

### Baseline Response
> {base_ans}

### Optimized Response
> {opt_ans}

---
*Report generated by `compare_optimizations.py`*
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[+] Comparison Reports Exported Successfully:")
    print(f"    - Markdown: {out_md}")
    print(f"    - JSON    : {out_json}\n")

    return comparison_payload


def main():
    parser = argparse.ArgumentParser(description="VidScribe Optimization Comparison Harness")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH), help="Path to baseline evaluation JSON report")
    parser.add_argument("--video", default=None, help="Video ID to evaluate (defaults to baseline video)")
    parser.add_argument("--question", default=None, help="Question to evaluate (defaults to baseline question)")
    parser.add_argument("--skip-ingestion", action="store_true", default=False, help="Skip Tier 1 ingestion benchmark (test QA only)")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    baseline_data = load_baseline_report(baseline_path)

    compare_benchmarks(
        baseline_data=baseline_data,
        video_id=args.video,
        question=args.question,
        skip_ingestion=args.skip_ingestion
    )


if __name__ == "__main__":
    main()
