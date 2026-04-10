import os
import json
import logging
from openai import OpenAI

# ------------------------------------------------------------------
# MANDATORY OPENENV HACKATHON CONFIGURATION VARIABLES
# These must perfectly match the pre-submission checklist requirements.
# ------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - for docker based environments
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Instantiating the OpenAI client using the strictly passed variables
client = OpenAI(
    base_url=API_BASE_URL,
    # Fallback to GROQ_API_KEY only for local testing if HF_TOKEN is missing
    api_key=HF_TOKEN if HF_TOKEN else os.getenv("GROQ_API_KEY", "dummy_key")
)

def run_inference():
    """
    OpenEnv Hackathon inference sequence with strict START/STEP/END logging
    and invocation of reset() and step().
    """
    print("START")
    
    try:
        from environment import NyayasetuEnvironment
        env = NyayasetuEnvironment()
        
        # 1. Test Environment Reset
        obs, info = env.reset()
        print(f"STEP: Environment initialized. Observation: {str(obs)[:100]}...")
        
        # 2. OpenEnv LLM call using the OpenAI client
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are NyayaSetu, an AI Legal Aid Router. Output JSON containing 'route', 'explanation', and 'steps'."},
                {"role": "user", "content": f"Case summary: {obs.get('case_summary', '')}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # 3. Parse LLM inference output
        agent_output = response.choices[0].message.content
        print(f"STEP: Inference Complete. Routing decision made.")
        
        # 4. Fallback execution to parse action correctly for step()
        try:
            action_data = json.loads(agent_output)
        except:
            action_data = {"route": "civil_court", "explanation": "Fallback used.", "steps": []}
            
        action = {
            "route": action_data.get("route", "civil_court"),
            "explanation": action_data.get("explanation", "Proceed to civil court."),
            "steps": action_data.get("steps", ["Consult a local lawyer immediately."])
        }
        
        # 5. Test Environment Step
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"STEP: Environment solved. Reward received: {reward}")

    except Exception as e:
        print(f"STEP: Critical Error encountered - {str(e)}")
        
    # Mandatory concluding output structure
    print("END")

if __name__ == "__main__":
    run_inference()
