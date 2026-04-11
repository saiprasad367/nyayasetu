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
    try:
        # ── Import environment & models (flat HF Space layout) ──────────────
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        from environment import NyayasetuEnvironment
        from models import LegalAidAction

        env = NyayasetuEnvironment()
    except Exception as exc:
        print(f"[END] task=NyayaSetu_Init score=0.0001 steps=0 error={exc}", flush=True)
        return

    # Run at least 3 tasks (iterations) to satisfy the validator condition
    for i in range(1, 4):
        task_name = f"NyayaSetu_Task_{i}"
        print(f"[START] task={task_name}", flush=True)

        try:
            # ── 1. Reset the environment ─────────────────────────────────────
            obs = env.reset()
            # Initial step after reset
            print("[STEP] step=0 reward=0.0", flush=True)

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
            
            # The validator demands the score is STRICTLY between 0 and 1, i.e., in (0, 1)
            clamped_reward = min(0.9999, max(0.0001, result_obs.reward))
            
            # Format required by OpenEnv Phase 2: [STEP] step=N reward=R
            print(f"[STEP] step=1 reward={clamped_reward:.4f}", flush=True)

            # Final record: [END] task=NAME score=S steps=N
            print(f"[END] task={task_name} score={clamped_reward:.4f} steps=1", flush=True)

        except Exception as exc:  # noqa: BLE001
            # If it fails, we provide a valid END block strictly within (0, 1)
            print(f"[END] task={task_name} score=0.0001 steps=0 error={exc}", flush=True)


if __name__ == "__main__":
    run_inference()
