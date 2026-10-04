#!/usr/bin/env python3
"""
Comprehensive benchmark of ReelScout pipeline with 15 Instagram reels.
Measures performance (time per reel) and cost (tokens/cost) with vision=always and text-llm=always modes.
"""

import json
import shutil
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from reelscout.pipeline import analyze
from reelscout.config import Settings


# Test URLs - 15 Instagram reels
TEST_REELS = [
    "https://www.instagram.com/reel/Dd7x6noPjAM/",
    "https://www.instagram.com/reels/Dd6VsR7PI5w/",
    "https://www.instagram.com/reels/Dd8s6U4ztQK/",
    "https://www.instagram.com/reels/Dd7sPgmD32w/",
    "https://www.instagram.com/reel/Ddoe6dpyyka/",
    "https://www.instagram.com/reels/DdToB05gH14/",
    "https://www.instagram.com/reel/DEXn8yPOO56/",
    "https://www.instagram.com/reel/DEK7TaDRVf3/",
    "https://www.instagram.com/reel/DEP_xWqKA62/",
    "https://www.instagram.com/reel/DZM8Q8KE-Vr/",
    "https://www.instagram.com/reel/Dd7r9fHFTr3/",
    "https://www.instagram.com/reel/Ddy8hUXyOln/",
    "https://www.instagram.com/reel/Daj9qR_sqKW/",
    "https://www.instagram.com/reel/Db_0MZyR8Ir/",
    "https://www.instagram.com/reel/Dd6THigT62v/",
]

# Token costs (per 1M tokens) - update based on your providers
# Based on .env: TEXT=groq (qwen/qwen3.8-27b free), VISION=xai (grok-3 paid), RESEARCH=groq (free)
TOKEN_COSTS = {
    "grok-3": {"input": 3.00, "output": 15.00},  # xAI Grok-3 pricing
    "qwen/qwen3.8-27b": {"input": 0.0, "output": 0.0},  # Groq free tier
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},  # Free tier
    "llama-3.3-70b": {"input": 0.0, "output": 0.0},  # Free tier
    "claude-haiku": {"input": 0.25, "output": 1.25},
    "gpt-4.1": {"input": 2.50, "output": 10.00},
}


def clear_cache():
    """Delete all cache directories."""
    cache_dir = Path(".cache")
    if cache_dir.exists():
        print(f"  Deleting cache directory: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)
        print("  Cache cleared", flush=True)
    else:
        print("  No cache directory found", flush=True)


def calculate_cost(usage: dict) -> tuple[float, dict]:
    """Calculate cost in USD based on token usage per model."""
    total_cost = 0.0
    cost_breakdown = {}
    
    for stage_name, stage_usage in usage.items():
        model = stage_usage.get("model", "")
        if not model:
            continue
        
        # Extract model name (remove provider prefix like "groq:" or "xai:")
        model_name = model.split(":")[-1] if ":" in model else model
        
        tokens_in = stage_usage.get("in", 0)
        tokens_out = stage_usage.get("out", 0)
        calls = stage_usage.get("calls", 1)
        
        costs = TOKEN_COSTS.get(model_name, {"input": 0.0, "output": 0.0})
        cost_in = (tokens_in / 1_000_000) * costs["input"]
        cost_out = (tokens_out / 1_000_000) * costs["output"]
        stage_cost = (cost_in + cost_out) * calls
        
        total_cost += stage_cost
        cost_breakdown[stage_name] = {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "calls": calls,
            "cost_usd": round(stage_cost, 6)
        }
    
    return round(total_cost, 6), cost_breakdown


def run_single_reel(source: str, settings: Settings, timeout_seconds: int = 120) -> dict:
    """Run a single reel analysis with timeout."""
    result_data = {
        "source": source,
        "success": False,
        "error": None,
        "elapsed_seconds": 0,
        "timings": {},
        "llm_usage": {},
        "llm_error": None,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_breakdown": {},
        "used_vision": False,
        "status": None,
        "summary": "",
        "confidence": None,
        "candidates": [],
        "resolution": {},
    }
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for timeout control
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyze, source, settings, "", None, "cookies.txt")
        
        try:
            r = future.result(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            
            # Extract detailed metrics from pipeline result
            timings = r.get("timings", {})
            llm_data = r.get("llm", {})
            usage = llm_data.get("usage", {})
            stages_log = llm_data.get("stages", [])
            llm_error = llm_data.get("error")
            
            # Calculate total tokens
            total_in = sum(stage.get("in", 0) for stage in stages_log)
            total_out = sum(stage.get("out", 0) for stage in stages_log)
            
            # Calculate cost
            total_cost, cost_breakdown = calculate_cost({s["stage"]: s for s in stages_log})
            
            # Get top candidate confidence
            cands = r.get("candidates", [])
            top_confidence = cands[0]["confidence"] if cands else None
            
            result_data.update({
                "success": True,
                "elapsed_seconds": round(elapsed, 2),
                "timings": timings,
                "llm_usage": usage,
                "llm_stages": stages_log,
                "llm_error": llm_error,
                "total_tokens_in": total_in,
                "total_tokens_out": total_out,
                "total_tokens": total_in + total_out,
                "cost_usd": total_cost,
                "cost_breakdown": cost_breakdown,
                "used_vision": r.get("used_vision", False),
                "status": r.get("status"),
                "summary": r.get("summary", ""),
                "confidence": top_confidence,
                "candidates": cands[:3],  # Top 3
                "resolution": r.get("resolution", {}),
            })
            
        except FutureTimeoutError:
            elapsed = time.time() - start_time
            result_data.update({
                "success": False,
                "error": "timeout",
                "elapsed_seconds": round(elapsed, 2),
            })
            print(f"  TIMEOUT after {elapsed:.1f}s", flush=True)
            
        except Exception as e:
            elapsed = time.time() - start_time
            result_data.update({
                "success": False,
                "error": str(e)[:200],
                "elapsed_seconds": round(elapsed, 2),
            })
            print(f"  ERROR: {e}", flush=True)
    
    return result_data


def print_reel_result(result: dict, index: int, total: int):
    """Print formatted result for a single reel."""
    print(f"\n[{index}/{total}] {result['source']}")
    print(f"  Vision: always | Text-LLM: always | Resolve: off")
    
    if result["success"]:
        print(f"  ✓ SUCCESS in {result['elapsed_seconds']}s")
        print(f"  Status: {result['status']}")
        print(f"  Summary: {result['summary'][:120]}...")
        print(f"  Top confidence: {result['confidence']}")
        print(f"  Used vision: {result['used_vision']}")
        print(f"  Tokens: {result['total_tokens']:,} (in: {result['total_tokens_in']:,}, out: {result['total_tokens_out']:,})")
        print(f"  Cost: ${result['cost_usd']:.6f}")
        
        # Stage timings
        timings = result.get("timings", {})
        if timings:
            print(f"  Stage timings:")
            for stage, t in timings.items():
                if stage != "total":
                    print(f"    {stage}: {t:.2f}s")
        
        # LLM stage details
        stages = result.get("llm_stages", [])
        if stages:
            print(f"  LLM stages:")
            for s in stages:
                print(f"    {s['stage']}: {s['model']} | in={s.get('in',0)} out={s.get('out',0)} calls={s.get('calls',1)}")
    else:
        print(f"  ✗ FAILED: {result['error']} (after {result['elapsed_seconds']}s)")


def generate_aggregate_summary(results: list) -> dict:
    """Generate aggregate summary from all results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    timeouts = [r for r in failed if r.get("error") == "timeout"]
    errors = [r for r in failed if r.get("error") != "timeout"]
    
    if successful:
        avg_time = sum(r["elapsed_seconds"] for r in successful) / len(successful)
        total_tokens = sum(r["total_tokens"] for r in successful)
        total_cost = sum(r["cost_usd"] for r in successful)
        avg_cost = total_cost / len(successful)
        vision_used = sum(1 for r in successful if r["used_vision"])
        
        # Average stage timings
        stage_timings_sum = {}
        stage_counts = {}
        for r in successful:
            for stage, t in r.get("timings", {}).items():
                if stage != "total":
                    stage_timings_sum[stage] = stage_timings_sum.get(stage, 0) + t
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        avg_stage_timings = {
            stage: round(stage_timings_sum[stage] / stage_counts[stage], 2)
            for stage in stage_timings_sum
        }
    else:
        avg_time = 0
        total_tokens = 0
        total_cost = 0
        avg_cost = 0
        vision_used = 0
        avg_stage_timings = {}
    
    return {
        "total_reels": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "timeouts": len(timeouts),
        "errors": len(errors),
        "success_rate": round(len(successful) / len(results) * 100, 1) if results else 0,
        "avg_time_seconds": round(avg_time, 2),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_reel_usd": round(avg_cost, 6),
        "vision_used_count": vision_used,
        "avg_stage_timings": avg_stage_timings,
        "failed_urls": [r["source"] for r in failed],
        "timeout_urls": [r["source"] for r in timeouts],
        "error_urls": [r["source"] for r in errors],
    }


def main():
    print("=" * 80)
    print("REELSCOUT COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(TEST_REELS)} reels")
    print(f"Configuration:")
    print(f"  --text-llm always")
    print(f"  --vision always")
    print(f"  --resolve off")
    print(f"  --cookies cookies.txt")
    print(f"  --summary")
    print(f"  Timeout: 120 seconds per reel")
    print(f"  Clear cache between runs: YES")
    print("=" * 80)
    
    # Build settings from env + CLI overrides
    settings = Settings.from_env()
    settings.text_policy = "always"
    settings.vision = "on"
    settings.resolve = "off"
    settings.verify = True
    
    # Verify cookies file exists
    cookies_path = Path("cookies.txt")
    if cookies_path.exists():
        print(f"✓ Cookies file found: {cookies_path}")
    else:
        print(f"⚠ Cookies file not found: {cookies_path}")
    
    results = []
    
    for i, reel in enumerate(TEST_REELS, 1):
        print(f"\n{'='*60}")
        print(f"Running reel {i}/{len(TEST_REELS)}")
        print(f"{'='*60}")
        
        # Clear cache before each run
        clear_cache()
        
        # Run the reel
        result = run_single_reel(reel, settings, timeout_seconds=120)
        results.append(result)
        
        # Print result
        print_reel_result(result, i, len(TEST_REELS))
        
        # Small delay between runs to avoid rate limits
        if i < len(TEST_REELS):
            print("  Waiting 3s before next run...")
            time.sleep(3)
    
    # Generate aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    
    summary = generate_aggregate_summary(results)
    
    print(f"Total reels tested: {summary['total_reels']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']} (timeouts: {summary['timeouts']}, errors: {summary['errors']})")
    print(f"Success rate: {summary['success_rate']}%")
    print(f"Average time: {summary['avg_time_seconds']}s")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"Avg cost per reel: ${summary['avg_cost_per_reel_usd']:.6f}")
    print(f"Vision used: {summary['vision_used_count']}/{summary['successful']} successful runs")
    
    if summary["avg_stage_timings"]:
        print(f"\nAverage stage timings:")
        for stage, t in sorted(summary["avg_stage_timings"].items()):
            print(f"  {stage}: {t:.2f}s")
    
    if summary["failed_urls"]:
        print(f"\nFailed URLs:")
        for url in summary["failed_urls"]:
            print(f"  - {url}")
    
    # Prepare final output
    output = {
        "benchmark_config": {
            "vision": "always",
            "text_llm": "always",
            "resolve": "off",
            "cookies": "cookies.txt",
            "summary": True,
            "timeout_per_reel_seconds": 120,
            "clear_cache_between_runs": True,
            "timestamp": datetime.now().isoformat(),
        },
        "per_reel_results": results,
        "aggregate_summary": summary,
    }
    
    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    return 0 if summary["successful"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())