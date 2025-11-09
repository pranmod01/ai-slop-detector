# Multi-Agent AI Slop Detector using Specialized SLMs on AWS Trainium
Four small language models (1-3B params each) working together to detect low-quality AI-generated content:
- Agent 1: Genericity Detector (spots vague, could-be-anything language)
- Agent 2: Substance Analyzer (measures information density vs fluff)
- Agent 3: Style Analyzer (identifies characteristic AI phrases and patterns)
- Agent 4: Repetition Detector (finds formulaic structure)

Each agent is fine-tuned with LoRA on a specific aspect, then ensemble voting produces final verdict. Project completed through [AWS Small Language Build Day Hackathon](https://app.agihouse.org/events/smalllanguagemodel-20251108)

## Why use SLMs?
- Interpretability: Users see exactly which signals fired (not just "AI detected")
- Robustness: Harder to fool multiple specialists than one generalist
- Efficiency: 4 small models cost less to run than 1 large model
- AI Safety: Transparent, auditable decisions for high-stakes content moderation
- Cost: ~10x cheaper than GPT-4 API for detection at scale

## Tech Stack
Infrastructure:
- AWS Trainium (trn1.2xlarge) - Purpose-built AI training chips
- Neuron SDK - Hardware optimization for efficient training

Models:
- Base: Qwen 2.5 (1.5B parameters)
-4 LoRA adapters - Parameter-efficient fine-tuning (~5MB each vs 1.5GB full model)

Dataset: HC3 (Human vs ChatGPT comparison corpus)

Framework:
- PyTorch + Transformers + PEFT
- LoRA for efficient specialization
- Ensemble voting for robust detection

