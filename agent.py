"""
NyayaSetu — Hour 4: AI Agent Development
=========================================
Contains:
  1. RuleBasedAgent  — keyword-matching baseline (target: ~50% accuracy)
  2. LegalAidAgent   — Groq LLM agent (Llama 3.1 8B) with few-shot prompting (target: >70%)

Usage:
    from agent import RuleBasedAgent, LegalAidAgent
    from models import LegalAidAction

    # Rule-based
    rule_agent = RuleBasedAgent()
    action = rule_agent.predict("My neighbor built fence on my land")

    # LLM agent (requires GROQ_API_KEY in environment)
    llm_agent = LegalAidAgent()
    action = llm_agent.predict({"case_summary": "...", "case_language": 0})
"""

import os
import json
import re
import time
import logging

from typing import Optional

# ---------------------------------------------------------------------------
# System Prompt (from PDF Section 2.4.4)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a legal aid assistant for rural Indian citizens.
Given a land dispute case summary, you must:
1. Classify the case type (boundary_dispute, inheritance_dispute, tenancy_issue, encroachment, sale_dispute, or loan_dispute)
2. Route to the correct legal forum — choose EXACTLY ONE from:
   - civil_court        (property title, ownership, encroachment, inheritance fights)
   - revenue_department (land records, mutation, tenancy eviction, map corrections)
   - arbitration        (disputes where both parties agree to out-of-court settlement)
   - consumer_court     (builder fraud, developer disputes, housing society issues)
   - criminal_court     (forgery, fraud, document theft, violence)
3. Generate simple step-by-step guidance in the same language as the case (Hindi if Hindi, English if English)

IMPORTANT routing rules:
- "boundary", "encroach", "fence", "wall on my land" → civil_court
- "tenant", "rent", "evict", "landlord-tenant", "rent hike" → revenue_department (unless illegal eviction → civil_court)
- "builder", "housing society", "developer", "flat not given" → consumer_court
- "will forged", "document forged", "FIR", "fraud", "criminal" → criminal_court
- "inheritance", "will", "legal heir", "father died" → civil_court
- "bank", "loan", "auction", "mortgage" → civil_court
- "mutation", "land records", "survey number", "patta" → revenue_department

Respond ONLY with valid JSON in this exact format (no extra text before or after):
{
  "case_type": "boundary_dispute",
  "route": "civil_court",
  "explanation": "This is a boundary dispute requiring...",
  "steps": [
    "File a civil suit under Order VII CPC",
    "Request court-appointed land surveyor",
    "Submit property title documents"
  ]
}

Be concise, accurate, and use simple language accessible to citizens with limited legal knowledge.
"""

# ---------------------------------------------------------------------------
# 1. BASELINE RULE-BASED AGENT (target: ~50% accuracy)
# ---------------------------------------------------------------------------
class RuleBasedAgent:
    """
    Simple keyword-matching agent as baseline.
    No API calls, no LLM — pure heuristics.
    """

    # Routing keywords per route (checked in priority order)
    ROUTING_RULES = [
        # consumer_court — check before civil (builder/developer specific)
        ("consumer_court", [
            "builder", "developer", "housing society", "flat not given",
            "registration refused", "advance paid", "amount paid", "refund",
            "plot allotment", "real estate"
        ]),
        # criminal_court — forgery / fraud / violence
        ("criminal_court", [
            "forged", "forgery", "fir", "fraud", "cheating", "criminal",
            "document theft", "fake certificate", "benami"
        ]),
        # revenue_department — tenancy / mutation / records
        ("revenue_department", [
            "tenant", "tenancy", "rent", "landlord", "evict", "rent control",
            "mutation", "patta", "survey number", "land records", "tahsildar",
            "revenue", "map correction", "khata", "adangal", "7/12",
            "किरायेदार", "किराया", "मकान मालिक", "नामांतरण"
        ]),
        # arbitration — settlement language
        ("arbitration", [
            "mediation", "arbitration", "settlement", "both parties agree",
            "mutual", "negotiate"
        ]),
        # civil_court — everything else (default for property disputes)
        ("civil_court", [
            "boundary", "fence", "wall", "encroach", "occupy", "trespass",
            "inherit", "inheritance", "legal heir", "will", "succession",
            "father died", "mother died", "partition", "share in property",
            "bank", "loan", "mortgage", "auction", "encumbrance",
            "sale", "purchase", "registration", "title", "ownership",
            "possession", "survey", "court",
            "सीमा", "विरासत", "उत्तराधिकार", "बैंक", "कब्जा", "विवाद"
        ]),
    ]

    # Case type classification keywords
    CASE_TYPE_RULES = [
        ("boundary_dispute",   ["boundary", "fence", "wall", "survey", "demarcation", "सीमा", "मेड़"]),
        ("inheritance_dispute",["inherit", "will", "succession", "legal heir", "father died", "mother died",
                                 "partition", "विरासत", "उत्तराधिकार", "वसीयत", "बंटवारा"]),
        ("tenancy_issue",      ["tenant", "tenancy", "rent", "landlord", "evict", "subletting",
                                 "किरायेदार", "किराया", "मकान मालिक"]),
        ("encroachment",       ["encroach", "occupied", "trespass", "illegal occupation", "shed on my land",
                                 "कब्जा", "अतिक्रमण"]),
        ("sale_dispute",       ["sale", "purchase", "builder", "developer", "double sale", "registration",
                                 "बिक्री", "रजिस्ट्री", "बिल्डर"]),
        ("loan_dispute",       ["loan", "bank", "mortgage", "auction", "moneylender", "interest",
                                 "बैंक", "ऋण", "साहूकार", "नीलामी"]),
    ]

    def _classify_case_type(self, text: str) -> str:
        lower = text.lower()
        for case_type, keywords in self.CASE_TYPE_RULES:
            if any(kw in lower for kw in keywords):
                return case_type
        return "boundary_dispute"  # fallback

    def _classify_route(self, text: str) -> str:
        lower = text.lower()
        for route, keywords in self.ROUTING_RULES:
            if any(kw in lower for kw in keywords):
                return route
        return "civil_court"  # default fallback

    def _generate_steps(self, route: str, case_type: str) -> list:
        """Generate contextual steps based on route and case type."""
        steps_map = {
            "civil_court": {
                "boundary_dispute":    ["File civil suit under Order VII CPC", "Request court-appointed land surveyor", "Submit chain of title documents"],
                "inheritance_dispute": ["File civil suit to establish legal heirship", "Submit death certificate and legal heir certificate", "Apply for interim injunction on property"],
                "encroachment":        ["File civil suit for recovery of possession", "Submit land ownership documents", "Apply for interim injunction"],
                "sale_dispute":        ["File civil suit for declaration of title", "Submit registered sale deed", "Apply for interim injunction against other party"],
                "loan_dispute":        ["File civil suit against the bank", "Submit loan repayment receipts", "Request stay on auction proceedings"],
                "tenancy_issue":       ["File civil suit for illegal eviction", "Submit valid rent agreement", "Request injunction against landlord"],
            },
            "revenue_department": {
                "boundary_dispute":    ["Apply to Tehsildar for map correction", "Submit both conflicting map copies", "Request revenue official inspection"],
                "inheritance_dispute": ["Apply for mutation at Tehsildar office", "Submit death certificate and legal heir documents", "Follow up on mutation order"],
                "tenancy_issue":       ["File eviction petition before Rent Controller", "Submit rent agreement and payment ledger", "Attend hearing for eviction order"],
                "sale_dispute":        ["Apply for mutation to Mandal Revenue Officer", "Submit registered sale deed", "Request correction in revenue records"],
                "encroachment":        ["File complaint with Tahsildar", "Submit land records", "Request field verification"],
                "loan_dispute":        ["File complaint with revenue authority", "Submit loan documents", "Request stay on attachment"],
            },
            "consumer_court": {
                "sale_dispute":        ["File consumer complaint before District Consumer Forum", "Submit payment receipts and sale agreement", "Seek registration or full refund with interest"],
                "tenancy_issue":       ["File consumer complaint", "Submit evidence of deficiency", "Claim compensation"],
                "default":             ["File consumer complaint", "Submit all payment receipts and agreements", "Seek refund or specific performance"],
            },
            "criminal_court": {
                "sale_dispute":        ["File FIR for cheating under IPC Section 420", "Submit forged document evidence", "Engage a criminal lawyer"],
                "inheritance_dispute": ["File FIR for forgery under IPC Section 467", "Submit original documents", "Report to police with evidence"],
                "default":             ["File FIR with local police station", "Submit evidence of fraud/forgery", "Engage criminal lawyer"],
            },
            "arbitration": {
                "default":             ["Agree on a neutral arbitrator with the other party", "Submit dispute documents to arbitrator", "Attend arbitration proceedings"],
            },
        }
        route_steps = steps_map.get(route, {})
        return route_steps.get(case_type, route_steps.get("default", [
            "Consult a local legal aid lawyer",
            "Gather all relevant land documents",
            "Approach the appropriate authority",
        ]))

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Hindi (Devanagari script)."""
        devanagari_count = sum(1 for ch in text if '\u0900' <= ch <= '\u097F')
        return "hindi" if devanagari_count > 5 else "english"

    def predict(self, case_summary: str) -> dict:
        """
        Predict routing and guidance for a given case summary.
        Returns dict matching LegalAidAction fields.
        """
        case_type = self._classify_case_type(case_summary)
        route = self._classify_route(case_summary)
        steps = self._generate_steps(route, case_type)
        lang = self._detect_language(case_summary)

        # Simple explanation templates
        route_labels = {
            "civil_court": "Civil Court" if lang == "english" else "सिविल कोर्ट",
            "revenue_department": "Revenue Department" if lang == "english" else "राजस्व विभाग",
            "consumer_court": "Consumer Court" if lang == "english" else "उपभोक्ता न्यायालय",
            "criminal_court": "Criminal Court (Police / FIR)" if lang == "english" else "पुलिस / FIR",
            "arbitration": "Arbitration / Mediation" if lang == "english" else "मध्यस्थता",
        }

        if lang == "english":
            explanation = (
                f"This appears to be a {case_type.replace('_', ' ')}. "
                f"Based on the details provided, you should approach the {route_labels[route]}. "
                f"This forum has jurisdiction to resolve your dispute and provide relief."
            )
        else:
            explanation = (
                f"यह मामला {case_type.replace('_', ' ')} से संबंधित प्रतीत होता है। "
                f"दिए गए विवरण के आधार पर, आपको {route_labels[route]} से संपर्क करना चाहिए।"
            )

        return {
            "case_type": case_type,
            "route": route,
            "explanation": explanation,
            "steps": steps,
        }


# ---------------------------------------------------------------------------
# 2. LLM AGENT — Groq + Llama 3.1 8B (target: >70% accuracy)
# ---------------------------------------------------------------------------
class LegalAidAgent:
    """
    LLM-powered agent using Groq API (Llama 3.1 8B Instant).
    Uses few-shot prompting for high accuracy legal case routing.
    Requires: GROQ_API_KEY environment variable.
    """

    VALID_ROUTES = {
        "civil_court", "revenue_department", "arbitration",
        "consumer_court", "criminal_court"
    }
    VALID_CASE_TYPES = {
        "boundary_dispute", "inheritance_dispute", "tenancy_issue",
        "encroachment", "sale_dispute", "loan_dispute"
    }

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment.\n"
                "Set it with: $env:GROQ_API_KEY='your-key-here'  (PowerShell)\n"
                "or add to .env file"
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.few_shot_examples = self._load_examples()
        self._rule_agent = RuleBasedAgent()  # Fallback

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("LegalAidAgent")

    def _load_examples(self) -> list:
        """Load few-shot examples from data directory."""
        # Resolve path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        examples_path = os.path.join(base_dir, "data", "few_shot_examples.json")

        if os.path.exists(examples_path):
            with open(examples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _build_few_shot_text(self) -> str:
        """Format few-shot examples into prompt text."""
        if not self.few_shot_examples:
            return ""

        parts = ["Here are examples of correct case routing:\n"]
        for i, ex in enumerate(self.few_shot_examples, 1):
            output_json = json.dumps(ex["output"], ensure_ascii=False, indent=2)
            parts.append(f"Example {i}:\nCase: {ex['input']}\nOutput:\n```json\n{output_json}\n```\n")

        return "\n".join(parts)

    def _build_prompt(self, case_summary: str) -> str:
        """Build the complete prompt with few-shot examples."""
        few_shot_text = self._build_few_shot_text()

        prompt = f"""{few_shot_text}
Now classify this new case and provide routing guidance:

Case: {case_summary}

Remember: Respond ONLY with valid JSON. No extra text.
"""
        return prompt

    def _parse_response(self, response_text: str) -> dict:
        """Extract and validate JSON from LLM response."""
        # Try to extract JSON block (handle markdown code fences)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        # Try direct JSON parse
        try:
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            obj_match = re.search(r'\{[\s\S]*\}', response_text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        required = ["case_type", "route", "explanation", "steps"]
        if not all(k in data for k in required):
            return None

        # Normalize route to valid values
        route = data.get("route", "").lower().replace(" ", "_").replace("-", "_")
        if route not in self.VALID_ROUTES:
            # Try to map common LLM outputs to valid routes
            route_map = {
                "civil": "civil_court",
                "revenue": "revenue_department",
                "consumer": "consumer_court",
                "criminal": "criminal_court",
                "police": "criminal_court",
                "mediation": "arbitration",
            }
            for key, val in route_map.items():
                if key in route:
                    route = val
                    break
            else:
                route = "civil_court"  # fallback
        data["route"] = route

        # Normalize case_type
        case_type = data.get("case_type", "").lower().replace(" ", "_")
        if case_type not in self.VALID_CASE_TYPES:
            case_type = "boundary_dispute"
        data["case_type"] = case_type

        # Ensure steps is a list
        if not isinstance(data.get("steps"), list):
            data["steps"] = [str(data["steps"])]

        return data

    def predict(self, observation: dict, max_retries: int = 3) -> dict:
        """
        Predict routing and guidance using Groq LLM.

        Args:
            observation: dict with 'case_summary' key (from env.reset())
            max_retries: number of retries on API failure

        Returns:
            dict with case_type, route, explanation, steps
        """
        case_summary = observation.get("case_summary", str(observation))
        prompt = self._build_prompt(case_summary)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,       # deterministic for routing
                    max_tokens=600,
                    response_format={"type": "json_object"},  # force JSON mode
                )

                raw_text = response.choices[0].message.content
                parsed = self._parse_response(raw_text)

                if parsed:
                    return parsed
                else:
                    self.logger.warning(f"Attempt {attempt+1}: Failed to parse LLM response — {raw_text[:100]}")

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    self.logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Attempt {attempt+1} failed: {error_msg}")
                    if attempt == max_retries - 1:
                        break

        # Fallback to rule-based agent if LLM fails
        self.logger.warning("LLM failed after all retries. Falling back to RuleBasedAgent.")
        return self._rule_agent.predict(case_summary)

    def predict_with_env_action(self, observation: dict):
        """
        Wrapper that returns a LegalAidAction object instead of dict.
        Compatible with the NyayaSetu OpenEnv environment.
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models import LegalAidAction

        result = self.predict(observation)
        return LegalAidAction(
            route=result["route"],
            explanation=result["explanation"],
            steps=result["steps"],
        )
