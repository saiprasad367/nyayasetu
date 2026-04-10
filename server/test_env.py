"""
Hour 3 Verification Script -- NyayaSetu OpenEnv Environment
Tests: reset(), step(), reward calculation, action/observation spaces
Expected output per the 8-Hour Plan:
  - reset() returns valid LegalAidObservation
  - step() returns (obs, reward, done, info) equivalent
  - Reward range is [0.0, 1.0]
  - Environment registered / importable correctly
"""

import sys
import os

# Allow running directly from the server/ directory OR from nyayasetu_env/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.nyayasetu_env_environment import NyayasetuEnvironment
from models import LegalAidAction

print("=" * 60)
print("  NyayaSetu -- Hour 3 Environment Verification")
print("=" * 60)

env = NyayasetuEnvironment()
print(f"\n[OK] Environment loaded. Dataset size: {len(env.data)} cases\n")

# ── TEST 1: reset() ──────────────────────────────────────────
obs = env.reset()
print("TEST 1: reset()")
print(f"   case_id    : {env.current_case['case_id']}")
print(f"   case_type  : {obs.case_type}")
print(f"   language   : {'Hindi' if obs.case_language == 1 else 'English'}")
print(f"   location   : {obs.location}")
print(f"   done       : {obs.done}")
print(f"   reward     : {obs.reward}")
print(f"   summary    : {obs.case_summary[:80]}...")
assert obs.done == False, "reset() must return done=False"
assert 0.0 <= obs.reward <= 1.0, "reward must be in [0, 1]"
print("   [PASS] reset() PASSED\n")

# ── TEST 2: step() with CORRECT route ───────────────────────
correct_route = env.current_case["correct_route"]
action_correct = LegalAidAction(
    route=correct_route,
    explanation=env.current_case["reasoning"],
    steps=env.current_case["steps"],
)
result_correct = env.step(action_correct)
print(f"TEST 2: step() with CORRECT route='{correct_route}'")
print(f"   reward     : {result_correct.reward}  (expected: ~0.6+)")
print(f"   done       : {result_correct.done}")
print(f"   metadata   : {result_correct.metadata}")
assert result_correct.done == True, "step() must return done=True"
assert result_correct.reward >= 0.6, f"Correct route should score >=0.6, got {result_correct.reward}"
print("   [PASS] step() CORRECT route PASSED\n")

# ── TEST 3: step() with WRONG route ─────────────────────────
env.reset()
correct_route_new = env.current_case["correct_route"]
wrong_routes = ["civil_court", "revenue_department", "arbitration", "consumer_court", "criminal_court"]
wrong_route = next(r for r in wrong_routes if r != correct_route_new)
action_wrong = LegalAidAction(
    route=wrong_route,
    explanation="wrong explanation",
    steps=["wrong step"],
)
result_wrong = env.step(action_wrong)
print(f"TEST 3: step() with WRONG route='{wrong_route}'")
print(f"   reward     : {result_wrong.reward}  (expected: <0.6)")
assert result_wrong.reward < 0.6, f"Wrong route should score <0.6, got {result_wrong.reward}"
print("   [PASS] step() WRONG route PASSED\n")

# ── TEST 4: reward range check across 5 random cases ────────
print("TEST 4: Reward range validation (5 random resets)")
for i in range(5):
    obs = env.reset()
    route = env.current_case["correct_route"]
    action = LegalAidAction(route=route, explanation="test", steps=["test step"])
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"   Case {i+1}: reward={result.reward}  route={route}")
print("   [PASS] Reward range [0.0, 1.0] PASSED\n")

# ── TEST 5: State tracking ───────────────────────────────────
print("TEST 5: State tracking")
obs = env.reset()
initial_steps = env.state.step_count
env.step(LegalAidAction(route="civil_court", explanation="test", steps=["step 1"]))
assert env.state.step_count == initial_steps + 1, "step_count not incrementing"
print(f"   step_count after 1 step: {env.state.step_count}")
print(f"   episode_id: {env.state.episode_id}")
print("   [PASS] State tracking PASSED\n")

print("=" * 60)
print("  ALL HOUR 3 VERIFICATION TESTS PASSED!")
print("  Environment is ready. Proceeding to Hour 4.")
print("=" * 60)