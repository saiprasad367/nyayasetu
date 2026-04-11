import os
import sys
import json
from openai import OpenAI

# ------------------------------------------------------------------
# MANDATORY OPENENV HACKATHON CONFIGURATION VARIABLES
# Strictly following the pre-submission checklist requirements.
# ------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set as a Space secret

# Optional — only used if from_docker_image() is called
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Instantiate the OpenAI-compatible client using the strictly defined variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

def run_inference():
    """
    OpenEnv Hackathon compliant inference sequence.
    Logs follow the mandatory START / STEP / END structured format.
    Calls environment reset() and step() as required by the validator.
    """
    print("START")

    try:
        # ── Import environment & models (flat HF Space layout) ──────────────
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        from environment import NyayasetuEnvironment
        from models import LegalAidAction

        env = NyayasetuEnvironment()

        # ── 1. Reset the environment ─────────────────────────────────────
        obs = env.reset()
        print(f"STEP: Environment reset complete. Case type: {obs.case_type}. Location: {obs.location}.")

        # ── 2. LLM inference call via OpenAI-compatible client ───────────
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are NyayaSetu, an AI Legal Aid Router for rural India. "
                        "Given a land dispute case summary, produce a JSON object with exactly "
                        "three keys: 'route' (one of: civil_court, revenue_department, "
                        "arbitration, consumer_court, criminal_court), 'explanation' (a clear "
                        "explanation for the citizen), and 'steps' (a list of 3-5 action steps)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Case summary: {obs.case_summary}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )

        agent_output = response.choices[0].message.content
        print("STEP: LLM inference complete. Parsing routing decision.")

        # ── 3. Parse LLM output and build a typed action ─────────────────
        try:
            action_data = json.loads(agent_output)
        except (json.JSONDecodeError, TypeError):
            action_data = {}

        action = LegalAidAction(
            route=action_data.get("route", "civil_court"),
            explanation=action_data.get(
                "explanation", "Please consult a local legal aid centre."
            ),
            steps=action_data.get(
                "steps", ["Visit the nearest District Legal Services Authority (DLSA)."]
            ),
        )

        # ── 4. Step the environment with the parsed action ────────────────
        result_obs = env.step(action)
        print(
            f"STEP: Environment step complete. "
            f"Reward: {result_obs.reward:.4f}. "
            f"Routed to: {action.route}."
        )

    except Exception as exc:  # noqa: BLE001
        print(f"STEP: Error during inference — {exc}")

    print("END")


if __name__ == "__main__":
    run_inference()
