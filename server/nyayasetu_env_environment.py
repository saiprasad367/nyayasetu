import random
import json
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import LegalAidAction, LegalAidObservation
except ImportError:
    from models import LegalAidAction, LegalAidObservation


# Resolve data path relative to this file so it works regardless of CWD
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class NyayasetuEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ✅ Load dataset — path resolved relative to this file
        data_path = os.path.join(_DATA_DIR, "train_cases.json")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.current_case = None

    # ✅ RESET — returns fresh observation from a random case
    def reset(self) -> LegalAidObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.current_case = random.choice(self.data)

        return LegalAidObservation(
            case_summary=self.current_case["case_summary"],
            case_language=0 if self.current_case["language"] == "english" else 1,
            case_type=self.current_case["case_type"],
            location=self.current_case["location"],
            reward=0.0,
            done=False,
        )

    # ✅ REWARD FUNCTION (PDF logic: routing 60% + explanation 25% + steps 15%)
    def calculate_reward(self, action: LegalAidAction, truth: dict) -> float:
        reward = 0.0

        # 1. Routing accuracy (60% weight)
        if action.route == truth["correct_route"]:
            reward += 0.6

        # 2. Explanation quality (25% weight) — keyword overlap heuristic
        truth_keywords = set(truth["reasoning"].lower().split())
        agent_keywords = set(action.explanation.lower().split())
        overlap = len(truth_keywords & agent_keywords)
        if truth_keywords:
            similarity = min(overlap / len(truth_keywords), 1.0)
            reward += 0.25 * similarity

        # 3. Step completeness (15% weight) — exact and partial step matches
        truth_steps = set(s.lower() for s in truth["steps"])
        agent_steps = set(s.lower() for s in action.steps)
        matched = len(truth_steps & agent_steps)
        if truth_steps:
            step_coverage = matched / len(truth_steps)
            reward += 0.15 * step_coverage

        return round(reward, 4)

    # ✅ STEP — takes an action, returns evaluated observation with reward
    def step(self, action: LegalAidAction) -> LegalAidObservation:
        self._state.step_count += 1

        truth = self.current_case
        reward = self.calculate_reward(action, truth)

        return LegalAidObservation(
            case_summary=truth["case_summary"],
            case_language=0 if truth["language"] == "english" else 1,
            case_type=truth["case_type"],
            location=truth["location"],
            reward=reward,
            done=True,
            metadata={
                "correct_route": truth["correct_route"],
                "reasoning": truth["reasoning"],
                "case_id": truth["case_id"],
            },
        )

    @property
    def state(self) -> State:
        return self._state