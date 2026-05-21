# AI Mongo Assistant

A simple AI chat application that uses LangChain to interpret natural language questions and query MongoDB.

## What it does
- Uses LangChain + Gemini/Gemma API (via Google Generative AI SDK) to plan MongoDB queries.
- Introspects the database at startup to discover collections and fields.
- Uses the first collection and first field as defaults when the question is ambiguous.
- Provides a FastAPI backend and a built-in HTML/CSS UI.

## Requirements
- Python 3.11+
- MongoDB Atlas connectivity (IP allowlist must allow your server IP)
- Gemini/Gemma API key

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file from `.env.example` and set values.

## Run locally
1. Start the API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Open http://localhost:8000

## Render deployment
Use a single FastAPI service:
- If Render is not honoring `runtime.txt`, deploy using the Dockerfile (Python 3.11).
- Start command (non-Docker): `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env vars: `MONGODB_URI`, `LLM_API_KEY`, `LLM_MODEL`

## Notes on MongoDB access
If you see connection timeouts, check MongoDB Atlas Network Access and allow the Render IP range or temporarily allow `0.0.0.0/0` for testing.
