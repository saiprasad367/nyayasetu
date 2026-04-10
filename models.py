from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict, Any


# ✅ ACTION SPACE (from PDF)
class LegalAidAction(Action):
    route: str = Field(..., description="Legal forum: civil_court / revenue_department / arbitration / consumer_court / criminal_court")
    explanation: str = Field(..., description="Explanation for the citizen in simple language")
    steps: List[str] = Field(..., description="Step-by-step guidance for the citizen")


# ✅ OBSERVATION SPACE (from PDF)
class LegalAidObservation(Observation):
    case_summary: str = Field(..., description="Case description provided by citizen")
    case_language: int = Field(..., description="0=English, 1=Hindi")
    case_type: str = Field(..., description="Type of dispute (boundary, inheritance, etc.)")
    location: str = Field(..., description="State/location of the case")
    reward: float = Field(default=0.0, description="Reward from environment (0.0-1.0)")
    done: bool = Field(default=False, description="Whether the episode is complete")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra information (correct_route, etc.)")


# ✅ Backward-compatible aliases (used in app.py / client.py scaffolding)
NyayasetuAction = LegalAidAction
NyayasetuObservation = LegalAidObservation