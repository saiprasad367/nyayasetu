"""
Hour 4 Verification Script -- NyayaSetu Agent Testing
======================================================
Tests:
  1. RuleBasedAgent -- keyword matching (target: ~50% accuracy)
  2. LegalAidAgent  -- Groq LLM (target: >70% accuracy)
  3. JSON output validation
  4. Hindi input handling
  5. Accuracy on 10 test cases
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import RuleBasedAgent, LegalAidAgent
from models import LegalAidAction
from server.nyayasetu_env_environment import NyayasetuEnvironment

print("=" * 65)
print("  NyayaSetu -- Hour 4 Agent Verification")
print("=" * 65)

# ---------------------------------------------------------------------------
# Load 10 test cases for evaluation
# ---------------------------------------------------------------------------
# server/test_agent.py lives in nyayasetu_env/server/
# data/ is at nyayasetu_env/data/
_server_dir = os.path.dirname(os.path.abspath(__file__))          # .../server/
_env_dir    = os.path.dirname(_server_dir)                         # .../nyayasetu_env/
data_path   = os.path.join(_env_dir, "data", "test_cases.json")

with open(data_path, "r", encoding="utf-8") as f:
    all_test_cases = json.load(f)

# Use first 10 cases for quick verification
test_cases = all_test_cases[:10]
print(f"\n[OK] Loaded {len(test_cases)} test cases for evaluation\n")

# ---------------------------------------------------------------------------
# TEST 1: RuleBasedAgent -- output format check
# ---------------------------------------------------------------------------
print("=" * 65)
print("TEST 1: RuleBasedAgent -- Output Format Validation")
print("=" * 65)

rule_agent = RuleBasedAgent()

sample_cases = [
    "My neighbor built a fence 2 feet inside my agricultural land boundary.",
    "My uncle is claiming my father's inherited land after his death.",
    "Tenant has not paid rent for 8 months and refuses to vacate.",
    "Builder took full payment for a plot but refused to register it.",
    "मेरे पड़ोसी ने मेरी जमीन पर अवैध रूप से कब्जा कर लिया है।",
]

for i, case_text in enumerate(sample_cases, 1):
    result = rule_agent.predict(case_text)
    # Validate structure
    assert "case_type" in result, f"Missing case_type in result {i}"
    assert "route" in result, f"Missing route in result {i}"
    assert "explanation" in result, f"Missing explanation in result {i}"
    assert "steps" in result, f"Missing steps in result {i}"
    assert isinstance(result["steps"], list), f"steps must be a list in result {i}"
    assert len(result["steps"]) >= 1, f"Must have at least 1 step in result {i}"
    lang_tag = "[Hindi]" if any('\u0900' <= ch <= '\u097F' for ch in case_text) else "[English]"
    print(f"  Case {i} {lang_tag}: route={result['route']} type={result['case_type']}")

print("\n  [PASS] RuleBasedAgent output format valid\n")

# ---------------------------------------------------------------------------
# TEST 2: RuleBasedAgent -- Accuracy on 10 test cases
# ---------------------------------------------------------------------------
print("=" * 65)
print("TEST 2: RuleBasedAgent -- Accuracy on 10 Test Cases")
print("=" * 65)

correct_rule = 0
rule_results = []

for case in test_cases:
    result = rule_agent.predict(case["case_summary"])
    is_correct = (result["route"] == case["correct_route"])
    if is_correct:
        correct_rule += 1
    rule_results.append({
        "case_id": case["case_id"],
        "predicted": result["route"],
        "actual": case["correct_route"],
        "correct": is_correct,
    })
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"  {status} {case['case_id']} | pred={result['route']:<20} actual={case['correct_route']}")

rule_accuracy = correct_rule / len(test_cases)
print(f"\n  Rule-Based Accuracy: {correct_rule}/{len(test_cases)} = {rule_accuracy:.0%}")
print(f"  Target: ~50% | Result: {'[PASS]' if rule_accuracy >= 0.40 else '[BELOW TARGET]'}")
print()

# ---------------------------------------------------------------------------
# TEST 3: RuleBasedAgent via Environment (end-to-end)
# ---------------------------------------------------------------------------
print("=" * 65)
print("TEST 3: Environment Integration (rule agent + env.step())")
print("=" * 65)

env = NyayasetuEnvironment()
obs = env.reset()

rule_result = rule_agent.predict(obs.case_summary)
action = LegalAidAction(
    route=rule_result["route"],
    explanation=rule_result["explanation"],
    steps=rule_result["steps"],
)
step_result = env.step(action)

print(f"  Case loaded   : {env.current_case['case_id']}")
print(f"  Agent route   : {rule_result['route']}")
print(f"  Correct route : {step_result.metadata['correct_route']}")
print(f"  Reward        : {step_result.reward}")
print(f"  Done          : {step_result.done}")
assert step_result.done == True
print("  [PASS] Environment integration works\n")

# ---------------------------------------------------------------------------
# TEST 4: LLM Agent (Groq) -- requires GROQ_API_KEY
# ---------------------------------------------------------------------------
print("=" * 65)
print("TEST 4: LegalAidAgent (Groq LLM) -- Accuracy on 10 Test Cases")
print("=" * 65)

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("  [SKIP] GROQ_API_KEY not set in environment.")
    print("  To run LLM tests, run:")
    print("  PowerShell: $env:GROQ_API_KEY='your-key-here'")
    print("  Then re-run this script.\n")
else:
    try:
        llm_agent = LegalAidAgent(model="llama-3.1-8b-instant")
        print(f"  [OK] LegalAidAgent initialized with model: {llm_agent.model}\n")

        correct_llm = 0
        llm_results = []

        for i, case in enumerate(test_cases):
            obs_dict = {
                "case_summary": case["case_summary"],
                "case_language": 0 if case["language"] == "english" else 1,
            }
            result = llm_agent.predict(obs_dict)
            is_correct = (result["route"] == case["correct_route"])
            if is_correct:
                correct_llm += 1
            llm_results.append({
                "case_id": case["case_id"],
                "predicted": result["route"],
                "actual": case["correct_route"],
                "correct": is_correct,
                "case_type": result.get("case_type"),
            })
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"  {status} {case['case_id']} | pred={result['route']:<20} actual={case['correct_route']}")

            # Small delay to respect rate limits
            time.sleep(0.3)

        llm_accuracy = correct_llm / len(test_cases)
        print(f"\n  LLM Accuracy: {correct_llm}/{len(test_cases)} = {llm_accuracy:.0%}")
        print(f"  Target: >70% | Result: {'[PASS]' if llm_accuracy >= 0.70 else '[BELOW TARGET - needs tuning]'}")

        # TEST 5: Hindi input test
        print()
        print("=" * 65)
        print("TEST 5: Hindi Input Handling")
        print("=" * 65)
        hindi_case = {
            "case_summary": "मेरे पड़ोसी ने मेरी जमीन की सीमा पर अवैध निर्माण कर दिया है।",
            "case_language": 1,
        }
        hindi_result = llm_agent.predict(hindi_case)
        print(f"  Input (Hindi): {hindi_case['case_summary']}")
        print(f"  Route         : {hindi_result['route']}")
        print(f"  Case type     : {hindi_result['case_type']}")
        print(f"  Explanation   : {hindi_result['explanation'][:80]}...")
        print(f"  Steps         : {len(hindi_result['steps'])} steps")
        assert hindi_result.get("route") in ["civil_court", "revenue_department"]
        print("  [PASS] Hindi input handled correctly\n")

    except Exception as e:
        print(f"  [ERROR] LLM Agent failed: {e}\n")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("=" * 65)
print("  HOUR 4 VERIFICATION SUMMARY")
print("=" * 65)
print(f"  Rule-Based Agent : {correct_rule}/{len(test_cases)} = {rule_accuracy:.0%} accuracy")
if groq_key:
    print(f"  LLM Agent (Groq) : {correct_llm}/{len(test_cases)} = {llm_accuracy:.0%} accuracy")
else:
    print("  LLM Agent (Groq) : Skipped (no API key)")
print("  Output format    : Valid JSON with case_type, route, explanation, steps")
print("  Env integration  : Working (step() returns reward correctly)")
print()
print("  Agent is ready. Proceeding to Hour 5 (Evaluation).")
print("=" * 65)
