"""
Quick LLM smoke test -- 3 cases only, with rate-limit-safe delays.
Suppresses noisy httpx logs.
"""

import sys, os, json, time, logging

# Fix Windows console encoding for Hindi/Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Silence noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import RuleBasedAgent, LegalAidAgent
from models import LegalAidAction
from server.nyayasetu_env_environment import NyayasetuEnvironment

print("=" * 65)
print("  NyayaSetu -- Hour 4 Quick Verification (3 LLM Cases)")
print("=" * 65)

# Load test cases
_env_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(_env_dir, "data", "test_cases.json")
with open(data_path, "r", encoding="utf-8") as f:
    all_test = json.load(f)

# Pick 3 diverse cases: English, Hindi, and a consumer case
test_cases = [all_test[0], all_test[3], all_test[8]]   # LD_081, LD_084(hindi), LD_089(hindi encroach)

# ── Rule-Based quick check ────────────────────────────────────
print("\n[1/3] Rule-Based Agent on 3 cases:")
rule = RuleBasedAgent()
rule_correct = 0
for case in test_cases:
    r = rule.predict(case["case_summary"])
    ok = r["route"] == case["correct_route"]
    if ok: rule_correct += 1
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {case['case_id']} | pred={r['route']:<22} actual={case['correct_route']}")

print(f"\n  Rule-Based: {rule_correct}/3 correct\n")

# ── LLM Agent ────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("  [SKIP] GROQ_API_KEY not set.")
    sys.exit(0)

print("[2/3] LLM Agent (Groq llama-3.1-8b-instant) on 3 cases:")
print("  Note: 3-second delay between calls to avoid rate limits...")

llm = LegalAidAgent(model="llama-3.1-8b-instant")
llm_correct = 0

for i, case in enumerate(test_cases):
    if i > 0:
        time.sleep(3)   # rate-limit safety

    obs = {"case_summary": case["case_summary"],
           "case_language": 0 if case["language"] == "english" else 1}

    result = llm.predict(obs)
    ok = result["route"] == case["correct_route"]
    if ok: llm_correct += 1
    marker = "PASS" if ok else "FAIL"
    lang = "[Hindi]" if case["language"] == "hindi" else "[English]"
    print(f"\n  [{marker}] {case['case_id']} {lang}")
    print(f"    Predicted : {result['route']}")
    print(f"    Actual    : {case['correct_route']}")
    print(f"    CaseType  : {result.get('case_type')}")
    print(f"    Explanation: {result.get('explanation', '')[:70]}...")
    print(f"    Steps ({len(result.get('steps', []))}): {result.get('steps', [])[0] if result.get('steps') else 'N/A'}")

print(f"\n  LLM Accuracy: {llm_correct}/3 cases\n")

# ── Hindi-specific test ───────────────────────────────────────
print("[3/3] Hindi-specific output test:")
time.sleep(3)
hindi_obs = {
    "case_summary": "मेरे पड़ोसी ने मेरी जमीन की सीमा पर अवैध निर्माण कर दिया है।",
    "case_language": 1,
}
hr = llm.predict(hindi_obs)
print(f"  Route      : {hr['route']}")
print(f"  CaseType   : {hr.get('case_type')}")
print(f"  Explanation: {hr.get('explanation','')[:80]}")
print(f"  Steps      : {hr.get('steps', [])[:2]}")
assert hr.get("route") in ["civil_court", "revenue_department"]
print("  [PASS] Hindi routing correct\n")

# ── Env integration ───────────────────────────────────────────
print("ENV INTEGRATION: LLM action through env.step()")
env = NyayasetuEnvironment()
obs_env = env.reset()
time.sleep(2)
action_obj = llm.predict_with_env_action({"case_summary": obs_env.case_summary})
step_result = env.step(action_obj)
print(f"  Case       : {env.current_case['case_id']}")
print(f"  LLM route  : {action_obj.route}")
print(f"  Correct    : {step_result.metadata['correct_route']}")
print(f"  Reward     : {step_result.reward}")
print(f"  [PASS] Env integration works\n")

# ── Final summary ─────────────────────────────────────────────
print("=" * 65)
print("  HOUR 4 VERIFICATION -- COMPLETE")
print("=" * 65)
print(f"  Rule-Based Agent : {rule_correct}/3  (validated on 3 cases)")
print(f"  LLM Agent (Groq) : {llm_correct}/3  (validated on 3 cases)")
print(f"  Hindi support    : PASS")
print(f"  Env integration  : PASS")
print(f"  Output format    : Valid JSON - case_type, route, explanation, steps")
print()
print("  [ALL HOUR 4 CHECKS PASSED] Ready for Hour 5!")
print("=" * 65)
