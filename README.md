---
title: NyayaSetu - Legal Aid AI for Rural India
emoji: ⚖️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered legal routing for 40M+ pending land disputes
---

# ⚖️ NyayaSetu — Legal Aid AI for Rural India

> **India's first AI-powered environment for land dispute resolution**
> Built for OpenEnv Hackathon 2026 | Targeting Top 1-2%

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.1%208B-orange)](https://groq.com)
[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Problem Statement

India has **40+ million pending land dispute cases**. Rural citizens — often with no legal literacy — don't know which forum to approach:

- ⚖️ Civil Court?
- 🏛️ Revenue Department?
- 🛒 Consumer Court?
- 🚨 Police / Criminal Court?

**NyayaSetu bridges this gap with explainable AI routing.**

---

## 🚀 Features

| Feature | Details |
|---|---|
| 🗂️ **6 Case Types** | Boundary, Inheritance, Tenancy, Encroachment, Sale, Loan |
| ⚖️ **Smart Routing** | Routes to correct legal forum (5 options) |
| 🌐 **Bilingual** | Full Hindi + English support |
| 📋 **Step-by-Step Guidance** | Actionable 3-5 step citizen guidance |
| 🤖 **Dual Agents** | Rule-Based (instant) + LLM Agent (Groq Llama 3.1 8B) |
| 📊 **Evaluation** | 85% routing accuracy on 20-case test set |
| 🔄 **OpenEnv API** | Full Environment with reset/step/reward |

---

## 📊 Evaluation Results

| Agent | Accuracy | F1 (Weighted) | Avg Reward |
|---|---|---|---|
| Rule-Based Baseline | 70% | 0.744 | 0.472 |
| **LLM Agent (Groq)** | **85%** ✅ | **0.872** | **0.615** |

**Target: >75% ✅ Achieved: 85%**

---

## 🏗️ Architecture

```
Citizen Input (Hindi/English case summary)
        ↓
OpenEnv Environment
  ├── Observation Space: case_summary, language, type, location
  └── Action Space: route, explanation, steps
        ↓
LLM Agent (Groq llama-3.1-8b-instant)
  ├── System Prompt with routing rules
  ├── 6 few-shot examples (1 per case type)
  └── Temperature: 0.1 (deterministic)
        ↓
Reward Calculation
  ├── Routing Accuracy: 60% weight
  ├── Explanation Quality: 25% weight
  └── Step Completeness: 15% weight
        ↓
Citizen Output: Forum + Explanation + Steps
```

---

## 🛠️ Tech Stack

- **Environment**: OpenEnv Core 0.2.3
- **LLM**: Groq (Llama 3.1 8B Instant) — Free tier
- **NLP**: Rule-based keyword routing (fallback)
- **UI**: Gradio 6.x
- **Data**: 100 real Indian land dispute cases (80 train / 20 test)
- **Deployment**: Hugging Face Spaces (CPU Free)

---

## 🚀 Quick Start (Local)

```bash
git clone https://huggingface.co/spaces/saiprasad25/nyayasetu-legal-env
cd nyayasetu-legal-env
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
python app.py
```

---

## 📁 File Structure

```
nyayasetu-legal-env/
├── app.py                    # Main Gradio interface (HF entry point)
├── agent.py                  # RuleBasedAgent + LegalAidAgent (Groq)
├── models.py                 # OpenEnv Action/Observation schemas
├── environment.py            # NyayaSetu OpenEnv environment
├── requirements.txt          # Dependencies
├── data/
│   ├── train_cases.json      # 80 training cases
│   ├── test_cases.json       # 20 test cases
│   └── few_shot_examples.json# 6 few-shot prompts
└── results/
    └── evaluation_metrics.json
```

---

## 👨‍💻 Author

**Saiprasad** | [@saiprasad25](https://huggingface.co/saiprasad25)
CMR College of Engineering & Technology | B.Tech CSE
📧 saiprasad2523@gmail.com

---

## 📜 License

MIT License — Free to use, modify, and deploy.

*Built for OpenEnv Hackathon 2026 — Targeting Top 1-2% finish*
