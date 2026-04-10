"""
NyayaSetu — Hour 5: Comprehensive Evaluation
=============================================
Runs BOTH RuleBasedAgent and LegalAidAgent on all 20 test cases.

Outputs (written to results/ folder):
  - evaluation_results.csv     ← predictions + reward per case
  - confusion_matrix.png       ← heatmap
  - accuracy_chart.png         ← per-category accuracy bar chart
  - evaluation_metrics.json    ← summary metrics (accuracy, F1, latency)
  - failure_analysis.txt       ← top 3 failure case debug report

Usage:
    cd nyayasetu_env
    python -X utf8 evaluate.py

With LLM (needs GROQ_API_KEY in environment):
    $env:GROQ_API_KEY="gsk_..."
    python -X utf8 evaluate.py
"""

import sys
import os
import json
import csv
import time
import logging
from datetime import datetime

# ── Fix Windows console encoding ─────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Silence HTTP noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# ── Add project root to path ──────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI popup)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)

from agent import RuleBasedAgent, LegalAidAgent
from models import LegalAidAction
from server.nyayasetu_env_environment import NyayasetuEnvironment

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH   = os.path.join(_ROOT, "data", "test_cases.json")
VALID_ROUTES = ["civil_court", "revenue_department", "arbitration",
                "consumer_court", "criminal_court"]

# ─────────────────────────────────────────────────────────────
# LOAD TEST SET
# ─────────────────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)

print("=" * 70)
print("  NyayaSetu - Hour 5: Comprehensive Evaluation & Testing")
print(f"  Test set: {len(TEST_CASES)} cases | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────
def run_evaluation(agent, agent_name: str, use_llm: bool = False) -> list:
    """
    Run agent on all test cases. Returns list of result dicts.
    Each result: case_id, case_type, language, correct_route,
                 predicted_route, correct, reward, latency_ms, explanation, steps
    """
    env = NyayasetuEnvironment()
    results = []

    print(f"\n  Running {agent_name} on {len(TEST_CASES)} test cases...")
    print(f"  {'Case ID':<10} {'Type':<22} {'Predicted':<22} {'Actual':<22} {'Result':<8} {'Reward'}")
    print(f"  {'-'*8} {'-'*21} {'-'*21} {'-'*21} {'-'*7} {'-'*6}")

    for i, case in enumerate(TEST_CASES):
        start_ms = time.time() * 1000

        # Build observation
        obs_dict = {
            "case_summary": case["case_summary"],
            "case_language": 0 if case["language"] == "english" else 1,
        }

        # Get prediction
        if use_llm:
            pred = agent.predict(obs_dict)
            if i < len(TEST_CASES) - 1:
                time.sleep(2.5)  # rate-limit safety for Groq free tier
        else:
            pred = agent.predict(case["case_summary"])

        latency = time.time() * 1000 - start_ms

        # Evaluate via environment
        # Manually set env current_case to evaluate reward
        env.current_case = case
        action = LegalAidAction(
            route=pred["route"],
            explanation=pred.get("explanation", ""),
            steps=pred.get("steps", []),
        )
        step_result = env.step(action)

        is_correct = pred["route"] == case["correct_route"]
        status = "PASS" if is_correct else "FAIL"

        result = {
            "case_id":        case["case_id"],
            "case_type":      case["case_type"],
            "language":       case["language"],
            "location":       case["location"],
            "correct_route":  case["correct_route"],
            "predicted_route":pred["route"],
            "correct":        is_correct,
            "reward":         step_result.reward,
            "latency_ms":     round(latency, 1),
            "explanation":    pred.get("explanation", "")[:120],
            "steps_count":    len(pred.get("steps", [])),
        }
        results.append(result)

        print(f"  {case['case_id']:<10} {case['case_type']:<22} {pred['route']:<22} "
              f"{case['correct_route']:<22} [{status}]   {step_result.reward:.3f}")

    return results


# ─────────────────────────────────────────────────────────────
# METRICS CALCULATION
# ─────────────────────────────────────────────────────────────
def calculate_metrics(results: list, agent_name: str) -> dict:
    """Compute all evaluation metrics from results list."""
    y_true = [r["correct_route"] for r in results]
    y_pred = [r["predicted_route"] for r in results]

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class F1 (weighted for class imbalance)
    labels = sorted(set(y_true + y_pred))
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)

    # Per-route accuracy
    per_route = {}
    for route in VALID_ROUTES:
        route_cases = [r for r in results if r["correct_route"] == route]
        if route_cases:
            route_acc = sum(1 for r in route_cases if r["correct"]) / len(route_cases)
            per_route[route] = {"count": len(route_cases), "accuracy": round(route_acc, 3)}

    # Average reward and latency
    avg_reward  = sum(r["reward"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    # Failure cases
    failures = [r for r in results if not r["correct"]]

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    metrics = {
        "agent":         agent_name,
        "total_cases":   len(results),
        "correct":       sum(1 for r in results if r["correct"]),
        "accuracy":      round(accuracy, 4),
        "f1_weighted":   round(f1_weighted, 4),
        "f1_macro":      round(f1_macro, 4),
        "avg_reward":    round(avg_reward, 4),
        "avg_latency_ms":round(avg_latency, 1),
        "failures":      len(failures),
        "per_route":     per_route,
        "report":        report,
    }
    return metrics, failures


# ─────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────
def save_csv(results_rule: list, results_llm: list):
    """Save combined evaluation results to CSV."""
    path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    fieldnames = ["case_id","case_type","language","location","correct_route",
                  "rule_predicted","rule_correct","rule_reward","rule_latency_ms",
                  "llm_predicted","llm_correct","llm_reward","llm_latency_ms"]

    # Align by case_id
    llm_by_id = {r["case_id"]: r for r in results_llm} if results_llm else {}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rr in results_rule:
            cid = rr["case_id"]
            lr  = llm_by_id.get(cid, {})
            writer.writerow({
                "case_id":          cid,
                "case_type":        rr["case_type"],
                "language":         rr["language"],
                "location":         rr["location"],
                "correct_route":    rr["correct_route"],
                "rule_predicted":   rr["predicted_route"],
                "rule_correct":     rr["correct"],
                "rule_reward":      rr["reward"],
                "rule_latency_ms":  rr["latency_ms"],
                "llm_predicted":    lr.get("predicted_route", "N/A"),
                "llm_correct":      lr.get("correct", "N/A"),
                "llm_reward":       lr.get("reward", "N/A"),
                "llm_latency_ms":   lr.get("latency_ms", "N/A"),
            })
    print(f"\n  [SAVED] evaluation_results.csv -> {path}")
    return path


# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX CHART
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(results: list, agent_name: str):
    """Generate and save confusion matrix heatmap."""
    y_true = [r["correct_route"] for r in results]
    y_pred = [r["predicted_route"] for r in results]
    labels = sorted(set(y_true + y_pred))
    short_labels = [l.replace("_", "\n") for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Predicted Route", fontsize=11)
    ax.set_ylabel("Actual Route", fontsize=11)
    ax.set_title(f"Confusion Matrix — {agent_name}", fontsize=13, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    fname = f"confusion_matrix_{agent_name.lower().replace(' ', '_')}.png"
    path  = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fname} -> {path}")
    return path


# ─────────────────────────────────────────────────────────────
# ACCURACY BAR CHART
# ─────────────────────────────────────────────────────────────
def plot_accuracy_chart(metrics_rule: dict, metrics_llm: dict = None):
    """Bar chart showing per-route accuracy for rule vs LLM agents."""
    routes = list(metrics_rule["per_route"].keys())
    rule_accs = [metrics_rule["per_route"].get(r, {}).get("accuracy", 0) for r in routes]

    x = range(len(routes))
    w = 0.35
    short_routes = [r.replace("_", "\n") for r in routes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - w/2 for i in x], rule_accs, w,
                   label="Rule-Based Agent", color="#4A90D9", alpha=0.85)

    if metrics_llm:
        llm_accs = [metrics_llm["per_route"].get(r, {}).get("accuracy", 0) for r in routes]
        bars2 = ax.bar([i + w/2 for i in x], llm_accs, w,
                       label="LLM Agent (Groq)", color="#27AE60", alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if metrics_llm:
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Target line
    ax.axhline(y=0.75, color="red", linestyle="--", linewidth=1.5, label="Target (75%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_routes, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Route Accuracy: Rule-Based vs LLM Agent\nNyayaSetu Legal Aid AI",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] accuracy_chart.png -> {path}")
    return path


# ─────────────────────────────────────────────────────────────
# FAILURE ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_failures(failures: list, agent_name: str, all_cases: list):
    """Analyze top 3 failure cases and generate debug report."""
    report_lines = [
        "=" * 70,
        f"FAILURE ANALYSIS REPORT — {agent_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total failures: {len(failures)} / {len(all_cases)}",
        "=" * 70,
    ]

    # Sort failures by reward (lowest first = worst misses)
    failures_sorted = sorted(failures, key=lambda x: x["reward"])

    for i, f in enumerate(failures_sorted[:3], 1):
        case_data = next((c for c in all_cases if c["case_id"] == f["case_id"]), {})
        report_lines += [
            f"\n--- Failure Case #{i}: {f['case_id']} ---",
            f"Case Type    : {f['case_type']}",
            f"Language     : {f['language']}",
            f"Summary      : {case_data.get('case_summary', '')[:120]}",
            f"Correct Route: {f['correct_route']}",
            f"Predicted    : {f['predicted_route']}",
            f"Reward       : {f['reward']}",
            f"Reasoning    : {case_data.get('reasoning', '')}",
            "",
            "ROOT CAUSE ANALYSIS:",
        ]

        # Auto-diagnose the failure
        cause, fix = diagnose_failure(f, case_data)
        report_lines.append(f"  Cause: {cause}")
        report_lines.append(f"  Fix  : {fix}")

    report_lines += [
        "\n" + "=" * 70,
        "SYSTEMATIC BIAS CHECK:",
    ]

    # Check for route-level bias
    from collections import Counter
    pred_counts = Counter(f["predicted_route"] for f in failures)
    actual_counts = Counter(f["correct_route"] for f in failures)
    for route, cnt in pred_counts.most_common():
        report_lines.append(f"  {cnt}x cases incorrectly routed TO {route}")
    for route, cnt in actual_counts.most_common():
        report_lines.append(f"  {cnt}x cases incorrectly routed AWAY FROM {route}")

    report_text = "\n".join(report_lines)

    path = os.path.join(RESULTS_DIR, "failure_analysis.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  [SAVED] failure_analysis.txt -> {path}")
    return report_text


def diagnose_failure(failure: dict, case_data: dict) -> tuple:
    """Diagnose why a routing failure happened and suggest a fix."""
    predicted = failure["predicted_route"]
    correct   = failure["correct_route"]
    summary   = case_data.get("case_summary", "").lower()
    lang      = failure["language"]

    if lang == "hindi" and predicted != correct:
        return (
            "Hindi case — agent may be defaulting to English routing logic",
            "Add more Hindi few-shot examples; improve system prompt with Hindi routing rules"
        )
    if correct == "consumer_court" and predicted == "civil_court":
        return (
            "Builder/developer dispute misclassified as general civil case",
            "Add keywords: 'builder', 'developer', 'advance paid', 'registration refused' → consumer_court"
        )
    if correct == "revenue_department" and predicted == "civil_court":
        return (
            "Revenue/record-level issue sent to civil court",
            "Add keywords: 'mutation', 'survey number', 'patta', 'tahsildar', 'revenue' → revenue_department"
        )
    if correct == "criminal_court" and predicted in ["civil_court", "revenue_department"]:
        return (
            "Criminal elements (forgery/fraud) not detected — routed to civil/revenue",
            "Add FIR/forgery/fraud as high-priority criminal_court trigger keywords"
        )
    if correct == "civil_court" and predicted == "revenue_department":
        return (
            "Civil dispute misclassified as revenue issue due to land record mention",
            "Refine: land records alone → revenue, but contested ownership + records → civil_court"
        )
    return (
        f"Incorrect routing: {predicted} instead of {correct}",
        "Review few-shot examples for this case type and add explicit example to prompt"
    )


# ─────────────────────────────────────────────────────────────
# SAVE METRICS JSON
# ─────────────────────────────────────────────────────────────
def save_metrics(metrics_rule: dict, metrics_llm: dict = None):
    data = {"rule_based": metrics_rule}
    if metrics_llm:
        data["llm_agent"] = metrics_llm

    # Remove non-serializable items
    for key in data:
        data[key].pop("report", None)

    path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] evaluation_metrics.json -> {path}")
    return path


# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
def print_summary(metrics: dict, label: str):
    print(f"\n  {'='*50}")
    print(f"  {label} — Results Summary")
    print(f"  {'='*50}")
    print(f"  Overall Accuracy : {metrics['accuracy']:.1%}  ({metrics['correct']}/{metrics['total_cases']})")
    print(f"  F1 (Weighted)    : {metrics['f1_weighted']:.3f}")
    print(f"  F1 (Macro)       : {metrics['f1_macro']:.3f}")
    print(f"  Avg Reward       : {metrics['avg_reward']:.3f}")
    print(f"  Avg Latency      : {metrics['avg_latency_ms']:.0f} ms/case")
    print(f"  Failures         : {metrics['failures']}")
    target_met = "YES" if metrics["accuracy"] >= 0.75 else "NO (below 75%)"
    print(f"  Target >75%      : {target_met}")
    print(f"\n  Per-Route Accuracy:")
    for route, stats in metrics["per_route"].items():
        bar = "#" * int(stats["accuracy"] * 20)
        print(f"    {route:<22} {stats['accuracy']:.0%}  [{bar:<20}]  ({stats['count']} cases)")
    print(f"\n  Classification Report:")
    for line in metrics["report"].split("\n")[:12]:
        print(f"    {line}")


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    groq_key = os.getenv("GROQ_API_KEY")

    # ── PHASE 1: Rule-Based Agent ─────────────────────────────
    print("\n" + "─"*70)
    print("  PHASE 1 — Rule-Based Agent Evaluation")
    print("─"*70)
    rule_agent   = RuleBasedAgent()
    results_rule = run_evaluation(rule_agent, "RuleBasedAgent", use_llm=False)
    metrics_rule, failures_rule = calculate_metrics(results_rule, "Rule-Based Agent")
    print_summary(metrics_rule, "Rule-Based Agent")

    # ── PHASE 2: LLM Agent (if API key present) ───────────────
    results_llm  = []
    metrics_llm  = None
    failures_llm = []

    if groq_key:
        print("\n" + "─"*70)
        print("  PHASE 2 — LLM Agent (Groq llama-3.1-8b-instant)")
        print("  Note: 2.5s delay between calls for rate limits...")
        print("─"*70)
        try:
            llm_agent   = LegalAidAgent(model="llama-3.1-8b-instant")
            results_llm = run_evaluation(llm_agent, "LLM Agent", use_llm=True)
            metrics_llm, failures_llm = calculate_metrics(results_llm, "LLM Agent")
            print_summary(metrics_llm, "LLM Agent (Groq)")
        except Exception as e:
            print(f"  [ERROR] LLM evaluation failed: {e}")
    else:
        print("\n  [SKIP] LLM Agent — GROQ_API_KEY not set")
        print("  Set key with: $env:GROQ_API_KEY='gsk_...'")

    # ── PHASE 3: Save outputs ─────────────────────────────────
    print("\n" + "─"*70)
    print("  PHASE 3 — Saving Outputs to results/ folder")
    print("─"*70)

    save_csv(results_rule, results_llm)
    save_metrics(metrics_rule, metrics_llm)
    plot_confusion_matrix(results_rule, "Rule-Based Agent")
    if results_llm:
        plot_confusion_matrix(results_llm, "LLM Agent")
    plot_accuracy_chart(metrics_rule, metrics_llm)

    # ── PHASE 4: Failure Analysis ─────────────────────────────
    print("\n" + "─"*70)
    print("  PHASE 4 — Failure Analysis (Top 3 failure cases)")
    print("─"*70)

    # Use LLM failures if available, else rule failures
    failures_to_analyze = failures_llm if failures_llm else failures_rule
    agent_label = "LLM Agent" if failures_llm else "Rule-Based Agent"

    if failures_to_analyze:
        report = analyze_failures(failures_to_analyze, agent_label, TEST_CASES)
        print("\n" + report[:1200])   # Print first 1200 chars
    else:
        print("  [PERFECT] No failures to analyze!")

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HOUR 5 EVALUATION — COMPLETE")
    print("=" * 70)
    rule_status = "PASS" if metrics_rule["accuracy"] >= 0.75 else "BELOW TARGET"
    print(f"  Rule-Based Agent : {metrics_rule['accuracy']:.0%} accuracy [{rule_status}]")
    if metrics_llm:
        llm_status = "PASS" if metrics_llm["accuracy"] >= 0.75 else "BELOW TARGET"
        print(f"  LLM Agent (Groq) : {metrics_llm['accuracy']:.0%} accuracy [{llm_status}]")
    else:
        print(f"  LLM Agent (Groq) : Skipped (no API key)")
    print(f"\n  Outputs saved to: {RESULTS_DIR}/")
    print(f"    - evaluation_results.csv")
    print(f"    - evaluation_metrics.json")
    print(f"    - confusion_matrix_*.png")
    print(f"    - accuracy_chart.png")
    print(f"    - failure_analysis.txt")
    print(f"\n  Ready for Hour 6 (Hugging Face Deployment)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
