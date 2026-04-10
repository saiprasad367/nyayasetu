"""
NyayaSetu OpenEnv Environment — HF Spaces version
Flat file structure (no server/ package needed).
"""

import random
import json
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import LegalAidAction, LegalAidObservation

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class NyayasetuEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        data_path = os.path.join(_DATA_DIR, "train_cases.json")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.current_case = None

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

    def calculate_reward(self, action: LegalAidAction, truth: dict) -> float:
        reward = 0.0
        if action.route == truth["correct_route"]:
            reward += 0.6
        truth_kw = set(truth["reasoning"].lower().split())
        agent_kw = set(action.explanation.lower().split())
        overlap = len(truth_kw & agent_kw)
        if truth_kw:
            reward += 0.25 * min(overlap / len(truth_kw), 1.0)
        truth_steps = set(s.lower() for s in truth["steps"])
        agent_steps = set(s.lower() for s in action.steps)
        if truth_steps:
            reward += 0.15 * (len(truth_steps & agent_steps) / len(truth_steps))
        return round(reward, 4)

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
            metadata={"correct_route": truth["correct_route"],
                      "reasoning": truth["reasoning"],
                      "case_id": truth["case_id"]},
        )

    @property
    def state(self) -> State:
        return self._state
