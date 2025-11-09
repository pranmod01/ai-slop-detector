# Multi-Agent AI Slop Detector using Specialized SLMs on AWS Trainium
Four small language models (1-3B params each) working together to detect low-quality AI-generated content:
- Agent 1: Genericity Detector (spots vague, could-be-anything language)
- Agent 2: Substance Analyzer (measures information density vs fluff)
- Agent 3: Style Analyzer (identifies characteristic AI phrases and patterns)
- Agent 4: Repetition Detector (finds formulaic structure)

Each agent is fine-tuned with LoRA on a specific aspect, then ensemble voting produces final verdict. Project completed through [AWS Small Language Build Day Hackathon](https://app.agihouse.org/events/smalllanguagemodel-20251108)

