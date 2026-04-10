# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nyayasetu Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LegalAidAction, LegalAidObservation


class NyayasetuEnv(
    EnvClient[LegalAidAction, LegalAidObservation, State]
):
    """
    Client for the NyayaSetu Legal Aid Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with NyayasetuEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.case_summary)
        ...
        ...     from .models import LegalAidAction
        ...     action = LegalAidAction(
        ...         route="civil_court",
        ...         explanation="This boundary dispute requires court intervention.",
        ...         steps=["File civil suit under Order VII CPC", "Submit land records"]
        ...     )
        ...     result = client.step(action)
        ...     print(result.reward)
    """

    def _step_payload(self, action: LegalAidAction) -> Dict:
        """
        Convert LegalAidAction to JSON payload for step message.

        Args:
            action: LegalAidAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "route": action.route,
            "explanation": action.explanation,
            "steps": action.steps,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LegalAidObservation]:
        """
        Parse server response into StepResult[LegalAidObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LegalAidObservation
        """
        obs_data = payload.get("observation", {})
        observation = LegalAidObservation(
            case_summary=obs_data.get("case_summary", ""),
            case_language=obs_data.get("case_language", 0),
            case_type=obs_data.get("case_type", ""),
            location=obs_data.get("location", ""),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", None),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
