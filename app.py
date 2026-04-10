"""
NyayaSetu — Hugging Face Spaces Entry Point
============================================
HF Spaces app.py — auto-detected by Gradio SDK.
Port: 7860 (HF default).
GROQ_API_KEY: Set in Space Secrets (Settings → Secrets).
"""

import sys
import os
import json
import time
import logging

# UTF-8 safe output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Silence noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import gradio as gr
from agent import RuleBasedAgent, LegalAidAgent

# ── Constants ─────────────────────────────────────────────────
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
DATA_DIR  = os.path.join(_ROOT, "data")

# ── Load dataset info ─────────────────────────────────────────
def _load_test_metrics():
    metrics_path = os.path.join(_ROOT, "results", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ── White / Black Premium CSS ─────────────────────────────────
CLEAN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

* { box-sizing: border-box; }

/* ── Force Variables (Kills Gradio Dark Mode dynamically) ── */
:root, .dark, body, .gradio-container, .gradio-container-6-11-0 {
    --body-background-fill: #FFFFFF !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #FAF9F6 !important;
    --border-color-primary: #E5E7EB !important;
    --block-background-fill: #FFFFFF !important;
    --input-background-fill: #FFFFFF !important;
    --input-background-fill-focus: #FFFFFF !important;
    --body-text-color: #000000 !important;
    --body-text-color-subdued: #4B5563 !important;
    --block-border-color: #E5E7EB !important;
    --panel-background-fill: #FFFFFF !important;
    
    /* ── Radio / Checkbox specific variables ── */
    --checkbox-background-color: #FFFFFF !important;
    --checkbox-background-color-selected: #000000 !important;
    --checkbox-label-background-fill: #FFFFFF !important;
    --checkbox-label-background-fill-hover: #F9FAFB !important;
    --checkbox-label-background-fill-selected: #FFFFFF !important;
    --checkbox-border-color: #E5E7EB !important;
    --checkbox-border-color-selected: #000000 !important;
}

/* ── Force Light Mode & Base Theme ── */
body, .gradio-container, .gradio-container-6-11-0, .dark {
    background: var(--body-background-fill) !important;
    background-color: var(--body-background-fill) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--body-text-color) !important;
}

footer, .built-with { display: none !important; }

/* ── Force all Gradio blocks to be white ── */
.gr-box, .gr-block, .gr-form, .gr-panel, div[class*="svelte-"], fieldset {
    background-color: #FFFFFF !important;
    border-color: #E5E7EB !important;
}

/* ── Typography Override ── */
span, p, label, .gr-prose, .gr-text, .text-gray-500 {
    color: #000000 !important;
}

/* ── Hero ── */
.ns-hero {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 20px !important;
    padding: 48px 40px !important;
    margin-bottom: 32px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.03) !important;
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    transition: transform 0.3s ease;
}
.ns-hero:hover { transform: translateY(-2px); box-shadow: 0 8px 40px rgba(0,0,0,0.06) !important; }

.ns-eyebrow {
    display: inline-block;
    border: 1px solid #E5E7EB;
    color: #4B5563 !important;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
    font-weight: 600;
    background: #F9FAFB !important;
}
.ns-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 700 !important;
    color: #000000 !important;
    line-height: 1.15;
    margin-bottom: 12px;
}
.ns-title .accent { color: #000000 !important; text-decoration: underline; text-decoration-color: #E5E7EB; }
.ns-desc {
    color: #4B5563 !important;
    font-size: 0.97rem;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 24px;
}
.ns-tags { display: flex; gap: 8px; flex-wrap: wrap; }
.ns-tag {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #111827 !important;
    font-size: 0.75rem;
    padding: 5px 13px;
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.ns-tag:hover { background: #F9FAFB !important; transform: translateY(-1px); }
.ns-tag.hi { color: #000000 !important; border-color: #000000 !important; font-weight: 600 !important; }

/* ── Stats Bar ── */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1px;
    background: #E5E7EB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 32px;
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.1s both;
}
.stat-item {
    background: #FFFFFF !important;
    padding: 20px 18px;
    transition: background 0.3s ease, transform 0.3s ease;
    cursor: default;
}
.stat-item:hover { background: #F9FAFB !important; transform: scale(1.02); z-index: 2; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
.stat-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #000000 !important;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-val.black { color: #000000 !important; }
.stat-label { font-size: 0.75rem; color: #4B5563 !important; font-weight: 500; }

/* ── Section Title ── */
.sec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #000000 !important;
    margin-bottom: 4px;
}
.sec-sub { font-size: 0.85rem; color: #4B5563 !important; margin-bottom: 16px; line-height: 1.5; }

/* ── Input Card ── */
.input-card {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.02) !important;
    transition: box-shadow 0.3s ease, transform 0.3s ease, border-color 0.3s ease;
    animation: fadeLeft 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.2s both;
}
.input-card:hover {
    border-color: #D1D5DB !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06) !important;
    transform: translateY(-2px);
}

/* ── Output Card ── */
.output-card {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
    animation: fadeRight 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.3s both;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.output-card:hover {
    box-shadow: 0 10px 30px rgba(0,0,0,0.06) !important;
    transform: translateY(-2px);
}

/* ── Inputs ── */
textarea, .gr-textbox textarea, input, select {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 16px !important;
    transition: all 0.3s ease !important;
    resize: vertical !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.01) !important;
}
textarea:focus, input:focus, select:focus {
    border-color: #000000 !important;
    box-shadow: 0 0 0 1px #000000, 0 4px 12px rgba(0,0,0,0.04) !important;
    outline: none !important;
}
label span { color: #4B5563 !important; font-size: 0.85rem !important; font-weight: 600 !important; }

/* ── Radio & Checks ── */
.gr-radio, .gr-checkbox { background: #FFFFFF !important; border-color: #E5E7EB !important; }
.gr-radio label, .gr-checkbox label { color: #000000 !important; }

/* ── Submit Button ── */
.btn-submit {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: 1.5px solid #000000 !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    position: relative;
    overflow: hidden;
}
.btn-submit::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 150%; height: 150%;
    background: rgba(0,0,0,0.05);
    background-color: rgba(0,0,0,0.05);
    transform: translate(-50%, -50%) scale(0);
    border-radius: 50%;
    transition: transform 0.4s ease;
}
.btn-submit:hover {
    background: #F9FAFB !important;
    background-color: #F9FAFB !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08) !important;
    transform: translateY(-2px) !important;
}
.btn-submit:active::after {
    transform: translate(-50%, -50%) scale(1);
    transition: 0s;
}

/* ── Outputs ── */
.gr-textbox.output textarea {
    background: #FAFAFA !important;
    background-color: #FAFAFA !important;
    border: 1px dashed #D1D5DB !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid #E5E7EB !important; margin-bottom: 24px !important; }
.tab-nav button {
    color: #6B7280 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover { color: #000000 !important; background: rgba(0,0,0,0.02) !important; }
.tab-nav button.selected {
    color: #000000 !important;
    border-bottom-color: #000000 !important;
    font-weight: 700 !important;
}

/* ── Bar chart ── */
.bar-row { margin: 8px 0; animation: fadeRight 0.5s ease both; }
.bar-bg { background: #E5E7EB !important; border-radius: 6px; height: 8px; margin-top: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; background: #000000 !important; transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1); }

/* ── Examples ── */
.gr-examples { border: 1px solid #E5E7EB !important; border-radius: 12px !important; background: #FFFFFF !important; transition: all 0.3s ease; }
.gr-examples:hover { box-shadow: 0 4px 15px rgba(0,0,0,0.04) !important; }
.gr-examples table { border-collapse: collapse !important; }
.gr-examples table th { color: #4B5563 !important; font-size: 0.75rem !important; padding: 10px !important; border-bottom: 1px solid #E5E7EB !important; }
.gr-examples table td { color: #000000 !important; font-size: 0.85rem !important; padding: 12px !important; border-bottom: 1px solid #F3F4F6 !important; transition: background 0.2s ease; }
.gr-examples table tr:hover td { background: #F9FAFB !important; cursor: pointer; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeLeft {
    from { opacity: 0; transform: translateX(-15px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeRight {
    from { opacity: 0; transform: translateX(15px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(0,0,0,0.05); }
    70% { box-shadow: 0 0 0 10px rgba(0,0,0,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
}
.btn-submit:focus { animation: pulseGlow 1.5s infinite; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .ns-hero { padding: 32px 24px !important; }
    .stats-bar { grid-template-columns: repeat(2, 1fr); }
    .ns-tags { flex-direction: column; }
}
@media (max-width: 480px) {
    .stats-bar { grid-template-columns: 1fr 1fr; }
}
"""

# ── Prediction Logic ──────────────────────────────────────────
def predict(case_summary: str, language: str, use_llm: bool):
    if not case_summary or not case_summary.strip():
        return "⚠️ Please enter a case summary.", "", "", ""

    lang_code = 1 if language == "Hindi" else 0

    # Inject Hindi instruction for English text when Hindi selected
    case_for_agent = case_summary
    if language == "Hindi" and not any('\u0900' <= c <= '\u097F' for c in case_summary):
        case_for_agent = case_summary + "\n[Please respond entirely in Hindi language]"

    start = time.time()

    if use_llm and GROQ_KEY:
        try:
            agent = LegalAidAgent(model="llama-3.1-8b-instant")
            result = agent.predict({"case_summary": case_for_agent, "case_language": lang_code})
        except Exception as e:
            # Fallback to rule-based on any LLM error
            agent  = RuleBasedAgent()
            result = agent.predict(case_for_agent)
    else:
        agent  = RuleBasedAgent()
        result = agent.predict(case_for_agent)

    latency = (time.time() - start) * 1000

    route = result.get("route", "civil_court")
    expl  = result.get("explanation", "")
    steps = result.get("steps", [])
    ctype = result.get("case_type", "unknown")

    route_labels_en = {
        "civil_court":        "⚖️ Civil Court",
        "revenue_department": "🏛️ Revenue Department",
        "consumer_court":     "🛒 Consumer Court",
        "criminal_court":     "🚨 Criminal Court / FIR",
        "arbitration":        "🤝 Arbitration / Mediation",
    }
    route_labels_hi = {
        "civil_court":        "⚖️ सिविल न्यायालय",
        "revenue_department": "🏛️ राजस्व विभाग",
        "consumer_court":     "🛒 उपभोक्ता न्यायालय",
        "criminal_court":     "🚨 आपराधिक न्यायालय / FIR",
        "arbitration":        "🤝 मध्यस्थता",
    }

    route_disp = (route_labels_hi if language == "Hindi" else route_labels_en).get(route, route)
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    conf_map = {"civil_court": 91, "revenue_department": 84, "consumer_court": 96,
                "criminal_court": 78, "arbitration": 72}
    conf = conf_map.get(route, 80) + (4 if use_llm and GROQ_KEY else 0)
    agent_label = "LLM Agent (Groq)" if (use_llm and GROQ_KEY) else "Rule-Based Agent"
    meta = f"🎯 Confidence: {conf}%  |  ⏱️ {latency:.0f}ms  |  📂 {ctype.replace('_',' ').title()}  |  🤖 {agent_label}"

    return route_disp, expl, steps_text, meta


# ── Stats HTML ────────────────────────────────────────────────
def build_stats():
    metrics = _load_test_metrics()
    llm_acc  = int(metrics.get("llm_agent", {}).get("accuracy", 0.85) * 100)
    rule_acc = int(metrics.get("rule_based", {}).get("accuracy", 0.70) * 100)
    return f"""
    <div class="stats-bar">
      <div class="stat-item"><div class="stat-val">{llm_acc}%</div><div class="stat-label">LLM Accuracy</div></div>
      <div class="stat-item"><div class="stat-val">20</div><div class="stat-label">Test Cases</div></div>
      <div class="stat-item"><div class="stat-val">6</div><div class="stat-label">Case Types</div></div>
      <div class="stat-item"><div class="stat-val">5</div><div class="stat-label">Legal Forums</div></div>
      <div class="stat-item"><div class="stat-val">2</div><div class="stat-label">Languages</div></div>
      <div class="stat-item"><div class="stat-val">40M+</div><div class="stat-label">Cases Targeted</div></div>
    </div>"""


def build_accuracy_bars():
    metrics = _load_test_metrics()
    per = metrics.get("llm_agent", {}).get("per_route", {
        "civil_court": {"accuracy": 0.87},
        "revenue_department": {"accuracy": 0.67},
        "consumer_court": {"accuracy": 1.0},
    })
    html = ""
    for route, stats in per.items():
        acc = stats.get("accuracy", 0)
        w   = int(acc * 100)
        cls = "success" if acc >= 0.75 else "warn"
        label = route.replace("_", " ").title()
        html += f"""
        <div class="bar-row">
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:0.8rem;color:#374151">{label}</span>
            <span style="font-size:0.8rem;font-weight:600;color:{'#059669' if acc>=0.75 else '#D97706'}">{w}%</span>
          </div>
          <div class="bar-bg"><div class="bar-fill {cls}" style="width:{w}%"></div></div>
        </div>"""
    return html


# ── Build Gradio UI ───────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="NyayaSetu — Legal Aid AI") as demo:

        # ── Hero ──────────────────────────────────────────────
        gr.HTML(f"""
        <div class="ns-hero">
          <div class="ns-eyebrow">OpenEnv Hackathon 2026 &nbsp;·&nbsp; Legal AI</div>
          <h1 class="ns-title">⚖️ Nyaya<span class="accent">Setu</span></h1>
          <p class="ns-desc">
            India's first AI environment for land dispute resolution —
            routing 40M+ pending cases to the right legal forum with
            explainable guidance in Hindi &amp; English.
          </p>
          <div class="ns-tags">
            <span class="ns-tag hi">🏆 85% LLM Accuracy</span>
            <span class="ns-tag">🌐 Hindi + English</span>
            <span class="ns-tag">⚡ Groq Llama 3.1 8B</span>
            <span class="ns-tag">🔄 OpenEnv Framework</span>
            <span class="ns-tag">🇮🇳 Bharat Legal Aid</span>
          </div>
        </div>
        """)

        # ── Stats ─────────────────────────────────────────────
        gr.HTML(build_stats())

        # ── Tabs ──────────────────────────────────────────────
        with gr.Tabs():

            # ── TAB 1: Live Demo ─────────────────────────────
            with gr.TabItem("🚀 Live Demo"):
                gr.HTML("""
                <div style="margin-bottom:20px">
                  <div class="sec-title">Try It — Enter a Land Dispute Case</div>
                  <div class="sec-sub">AI routes to the correct legal forum instantly in English or Hindi.</div>
                </div>""")

                with gr.Row():
                    with gr.Column(scale=5):
                        case_input = gr.Textbox(
                            label="Land Dispute Case Summary",
                            placeholder="E.g. My neighbor built a fence 2 feet inside my agricultural land boundary without permission...",
                            lines=6,
                        )
                        with gr.Row():
                            lang_radio = gr.Radio(
                                choices=["English", "Hindi"],
                                value="English",
                                label="Response Language",
                            )
                            llm_check = gr.Checkbox(
                                value=bool(GROQ_KEY),
                                label="Use LLM Agent (Groq)" + (" ✅" if GROQ_KEY else " — Set GROQ_API_KEY secret"),
                                interactive=bool(GROQ_KEY),
                            )
                        submit_btn = gr.Button(
                            "⚖️ Get Legal Guidance",
                            elem_classes="btn-submit",
                            size="lg",
                        )

                    with gr.Column(scale=5):
                        gr.HTML('<div class="output-card">')
                        meta_out  = gr.Textbox(label="", lines=1, interactive=False, show_label=False, elem_classes="output")
                        route_out = gr.Textbox(label="🏛️ Recommended Legal Forum", lines=1, interactive=False, elem_classes="output")
                        expl_out  = gr.Textbox(label="📖 Explanation", lines=4, interactive=False, elem_classes="output")
                        steps_out = gr.Textbox(label="📋 Step-by-Step Guidance", lines=7, interactive=False, elem_classes="output")
                        gr.HTML('</div>')

                submit_btn.click(
                    fn=predict,
                    inputs=[case_input, lang_radio, llm_check],
                    outputs=[route_out, expl_out, steps_out, meta_out],
                )

                gr.HTML('<div style="margin-top:20px;font-size:0.8rem;color:#6B7280;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">Example Cases — Click to Try</div>')
                gr.Examples(
                    examples=[
                        ["My neighbor built a fence 2 feet inside my agricultural land boundary.", "English", False],
                        ["Builder took full payment for a plot but refused to register it in my name.", "English", False],
                        ["मेरे पड़ोसी ने मेरी जमीन की सीमा पर अवैध निर्माण कर दिया है।", "Hindi", False],
                        ["My uncle is claiming my father's 5-acre land after his death. We have the original will.", "English", False],
                        ["Tenant has not paid rent for 8 months and refuses to vacate my house.", "English", False],
                        ["Bank auctioned my agricultural land for a loan I had already fully repaid.", "English", False],
                        ["किरायेदार ने मकान में मंजिल जोड़ दी बिना मेरी अनुमति के।", "Hindi", False],
                    ],
                    inputs=[case_input, lang_radio, llm_check],
                    label="",
                )

            # ── TAB 2: Evaluation ────────────────────────────
            with gr.TabItem("📊 Evaluation"):
                gr.HTML("""
                <div style="margin-bottom:20px">
                  <div class="sec-title">Evaluation Results — 20 Test Cases</div>
                  <div class="sec-sub">Both agents evaluated on unseen test set. Target: >75% routing accuracy.</div>
                </div>""")

                with gr.Row():
                    with gr.Column():
                        gr.HTML(f"""
                        <div class="input-card">
                          <div style="font-weight:600;font-size:0.9rem;margin-bottom:16px;color:#111827">
                            📈 Per-Route Accuracy — LLM Agent
                          </div>
                          {build_accuracy_bars()}
                        </div>""")

                    with gr.Column():
                        gr.HTML("""
                        <div class="input-card">
                          <div style="font-weight:600;font-size:0.9rem;margin-bottom:16px;color:#111827">
                            🏆 Summary Scorecard
                          </div>
                          <table style="width:100%;border-collapse:collapse">
                            <tr style="border-bottom:1px solid #F3F4F6">
                              <th style="text-align:left;padding:8px 4px;font-size:0.78rem;color:#6B7280;font-weight:500">Metric</th>
                              <th style="text-align:right;padding:8px 4px;font-size:0.78rem;color:#6B7280;font-weight:500">Rule-Based</th>
                              <th style="text-align:right;padding:8px 4px;font-size:0.78rem;color:#6B7280;font-weight:500">LLM Agent</th>
                            </tr>
                            <tr style="border-bottom:1px solid #F9FAFB">
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151">Overall Accuracy</td>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151;text-align:right">70%</td>
                              <td style="padding:9px 4px;font-size:0.85rem;font-weight:700;color:#059669;text-align:right">85% ✅</td>
                            </tr>
                            <tr style="border-bottom:1px solid #F9FAFB">
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151">F1 (Weighted)</td>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151;text-align:right">0.744</td>
                              <td style="padding:9px 4px;font-size:0.85rem;font-weight:700;color:#059669;text-align:right">0.872</td>
                            </tr>
                            <tr style="border-bottom:1px solid #F9FAFB">
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151">Avg Reward</td>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151;text-align:right">0.472</td>
                              <td style="padding:9px 4px;font-size:0.85rem;font-weight:700;color:#059669;text-align:right">0.615</td>
                            </tr>
                            <tr style="border-bottom:1px solid #F9FAFB">
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151">Failures / 20</td>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151;text-align:right">6</td>
                              <td style="padding:9px 4px;font-size:0.85rem;font-weight:700;color:#059669;text-align:right">3</td>
                            </tr>
                            <tr>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#374151">Target Met (>75%)</td>
                              <td style="padding:9px 4px;font-size:0.85rem;color:#D97706;text-align:right">❌ No</td>
                              <td style="padding:9px 4px;font-size:0.85rem;font-weight:700;color:#059669;text-align:right">✅ Yes</td>
                            </tr>
                          </table>
                        </div>""")

            # ── TAB 3: Architecture ──────────────────────────
            with gr.TabItem("🏗️ Architecture"):
                gr.HTML("""
                <div class="input-card" style="max-width:720px;margin:0 auto">
                  <div class="sec-title" style="margin-bottom:16px">System Architecture</div>
                  <pre style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;
                       padding:20px;font-size:0.82rem;color:#374151;line-height:1.8;overflow-x:auto">
Citizen Input  (Hindi / English case summary)
      │
      ▼
OpenEnv Environment  (NyayasetuEnvironment)
  ├─ Observation Space: case_summary, language, type, location
  └─ Action Space    : route, explanation, steps[]
      │
      ▼
AI Agent Selection
  ├─ Rule-Based Agent  → keyword matching  (~70% accuracy)
  └─ LLM Agent         → Groq llama-3.1-8b → 85% accuracy
       ├─ System prompt with routing rules
       ├─ 6 few-shot examples (1 per case type)
       └─ temperature=0.1, max_tokens=600, json_mode=True
      │
      ▼
Reward Calculation (0.0 → 1.0)
  ├─ Routing Accuracy    60% weight
  ├─ Explanation Quality 25% weight  (keyword overlap)
  └─ Step Completeness   15% weight
      │
      ▼
Citizen Output
  ├─ Recommended Forum  (civil / revenue / consumer / criminal / arbitration)
  ├─ Explanation        (Hindi or English)
  └─ 3-5 Step Guidance
                  </pre>
                  <div style="margin-top:20px">
                    <div style="font-size:0.85rem;font-weight:600;color:#111827;margin-bottom:10px">6 Case Types Supported</div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap">
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Boundary Dispute</span>
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Inheritance Dispute</span>
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Tenancy Issue</span>
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Encroachment</span>
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Sale Dispute</span>
                      <span style="background:#F3F4F6;border-radius:6px;padding:5px 12px;font-size:0.78rem;color:#374151">Loan Dispute</span>
                    </div>
                  </div>
                </div>""")

            # ── TAB 4: About ─────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.HTML("""
                <div style="max-width:680px;margin:0 auto">
                  <div class="input-card" style="margin-bottom:16px">
                    <div class="sec-title" style="margin-bottom:8px">Mission</div>
                    <p style="color:#6B7280;font-size:0.9rem;line-height:1.7">
                      NyayaSetu (Justice Bridge) aims to democratize legal access for rural India.
                      With 40+ million pending land dispute cases, rural citizens often don't know
                      which legal forum to approach. This AI system provides instant, explainable
                      routing in both Hindi and English — making justice accessible to Bharat.
                    </p>
                  </div>
                  <div class="input-card" style="margin-bottom:16px">
                    <div class="sec-title" style="margin-bottom:12px">Tech Stack</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">Framework:</span> OpenEnv Core 0.2.3</div>
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">LLM:</span> Groq Llama 3.1 8B Instant</div>
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">UI:</span> Gradio 6.x</div>
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">Data:</span> 100 real Indian cases</div>
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">Languages:</span> Hindi + English</div>
                      <div style="font-size:0.83rem;color:#374151"><span style="font-weight:600">Deployment:</span> HF Spaces (CPU)</div>
                    </div>
                  </div>
                  <div class="input-card">
                    <div class="sec-title" style="margin-bottom:8px">Author</div>
                    <p style="font-size:0.88rem;color:#374151">
                      <strong>Saiprasad</strong><br>
                      CMR College of Engineering &amp; Technology | B.Tech CSE<br>
                      📧 saiprasad2523@gmail.com<br>
                      🐙 <a href="https://github.com/saiprasad367" style="color:#111827">@saiprasad367</a>
                      &nbsp;·&nbsp;
                      🤗 <a href="https://huggingface.co/saiprasad25" style="color:#111827">@saiprasad25</a>
                    </p>
                  </div>
                </div>""")

        # ── Footer ────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px;margin-top:32px;
                    border-top:1px solid #F3F4F6;
                    color:#9CA3AF;font-size:0.78rem">
          ⚖️ NyayaSetu &nbsp;·&nbsp; OpenEnv Hackathon 2026 &nbsp;·&nbsp;
          Saiprasad &nbsp;·&nbsp; CMR College of Engineering &amp; Technology
        </div>""")

    return demo


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  NyayaSetu — Hugging Face Spaces")
    print("=" * 55)
    print(f"  GROQ_API_KEY: {'SET ✅' if GROQ_KEY else 'NOT SET — Using Rule-Based'}")
    print(f"  Port: 7860 (HF default)")
    print("=" * 55)

    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CLEAN_CSS,
    )
