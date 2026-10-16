# Agentic RAG System

This repository implements an agentic Retrieval-Augmented Generation (RAG) system.

Conventions
- Tests are required for core logic. Every PR that adds or changes behavior must include or update tests.
- Do not commit secrets or API keys into the repository. Use environment variables (e.g., OPENAI_API_KEY) or a secrets manager.
- Run linting (black + flake8) and tests locally before opening a PR.
- Work in small commits and open a PR for each working slice (feature / bugfix / doc).
- Before any PR that touches the Web UI, run the test-app skill documented in .agents/skill/test-app/ and attach a short video recording of the manual smoke test.
- Use the .agents/skill/test-app/run_test.sh script to automate verification.

Project layout (recommended)
- src/                # Application source code
  - ingest.py         # Document ingestion utilities
  - retrieval.py      # Retrieval agent and vector-store wrapper
  - webui/            # Streamlit frontend and small frontend assets
- tests/              # Pytest tests
- data/               # Example documents (not checked in if large)
- .agents/skill/      # Local test skills and automation
- AGENTS.md           # This file
- requirements.txt    # Dependencies

Branching & PR rules
- Create feature branches named like `feature/<short-desc>` or `fix/<short-desc>`.
- Rebase or merge the main branch regularly to keep PRs small.

CI / Quality
- We expect CI to run pytest and linting.
- Aim for clear testable functions; avoid hard-to-test side effects.

Contact
- Repository owner: narayanapudooru-a11y
