# test-app skill

This skill documents exactly how to install, run, and test the app locally. Use it before creating PRs that change behavior, and always use it before opening PRs that touch the UI.

Prerequisites
- Python 3.9+ (3.10+ recommended)
- Git
- An OpenAI API key set in the environment: OPENAI_API_KEY

Quick install

1. Clone the repository and switch to the branch you want to test:

   git clone https://github.com/narayanapudooru-a11y/Agentic-RAG-System.git
   cd Agentic-RAG-System

2. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\\Scripts\\activate    # Windows (PowerShell)

3. Install dependencies:

   pip install -r requirements.txt

Run unit tests

- Run the test suite with pytest. The test runner will exit with non-zero on failure.

   pytest -q

Run the web UI (smoke test)

- Start a Streamlit app (the repository's Streamlit entrypoint is expected at src/webui/app.py). If the file is missing, the smoke test will fail and that's intended.

   streamlit run src/webui/app.py --server.port 8501 &

- Wait a few seconds for the server to start, then validate the HTTP response:

   curl -sS --fail http://localhost:8501/ > /dev/null

- Stop the Streamlit server (the run_test.sh script will handle cleanup).

Automated test runner (run_test.sh)

This repository includes a helper that runs the tests and performs a minimal smoke test of the Streamlit UI. Use it before opening UI PRs.

- Make sure OPENAI_API_KEY is exported in your shell.
- Run the script (it will run pytest, start Streamlit, curl the root page, and exit non-zero on errors):

   bash .agents/skill/test-app/run_test.sh

Behavior and expectations
- All tests must pass (pytest exit code 0).
- The smoke test expects that streamlit serves an HTML page on port 8501. If the app is not present, the smoke test fails.

Recording manual test video

When a PR touches the UI, record a short screen capture (~30s) showing:
- Installation of requirements (pip install -r requirements.txt)
- Running the Streamlit app
- Uploading a small document (PDF or text) in the UI and asking a simple question
- The agent responding with an answer

Attach that video to the PR.

