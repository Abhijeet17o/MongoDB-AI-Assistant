# Technical Overview - MongoDB AI Assistant

## Summary
This project is a FastAPI-based AI assistant that translates natural language questions into MongoDB queries and returns human-readable answers. It uses a Gemini/Gemma model for query planning, a MongoDB driver for execution, and a single-page HTML/CSS/JS UI for interaction.

## Implementation
- Natural language question answering over a MongoDB Atlas dataset.
- LLM-driven query planning with schema-aware validation.
- Multi-step query decomposition for stacked questions.
- Safe query execution (limited results, field projection, redaction).
- A human-friendly UI with hierarchical details lists (no raw JSON).
- Deployment support for Render with Docker (Python 3.11) and health endpoints.

## Tech Stack (What is used for what)
- Python 3.11: runtime for the backend.
- FastAPI: HTTP API, routing, and validation.
- Uvicorn: ASGI server.
- MongoDB / PyMongo: database access and query execution.
- Gemini (google-generativeai): LLM used to build query plans.
- LangChain Core: prompt building + structured output parsing.
- HTML/CSS/JS: single-page UI served by FastAPI static files.
- Docker: consistent deployment runtime on Render.

## Architecture (High Level)
1. UI sends a user question to POST /chat.
2. Backend loads schema snapshot (collections + sample fields).
3. LLM produces one or more query plans (JSON).
4. Backend validates and normalizes the plan(s).
5. Queries are executed with PyMongo.
6. Response is summarized and sent to the UI.
7. UI formats results as a readable list with sections.

## Backend Components

### API Layer
- app/main.py
  - GET /: serves the UI (static HTML).
  - GET /health: basic liveness check.
  - GET /health/db: DB diagnostics (schema + error if any).
  - POST /chat: main QA endpoint.

### DB Layer
- app/db.py
  - Connects to MongoDB using MONGODB_URI.
  - If URI has no default DB name, uses first non-system DB.
  - Builds a schema snapshot:
    - collections
    - fields_by_collection (from a sample doc)
    - primary_collection / primary_field / first_word

### Agent / Planning Layer
- app/agent.py
  - Uses a structured QueryPlan model.
  - Supports multi-step planning (MultiQueryPlan) for stacked questions.
  - Applies validation using fields_by_collection.
  - Refines once if the plan fails or returns no data.
  - Executes count, sum, top, find actions.

## Query Flow in Detail
1. Schema snapshot is loaded at startup and on demand.
2. User question is sent to the LLM for planning.
3. LLM returns a plan (or multiple plans for multi-asks).
4. The plan is normalized:
   - fills missing collection
   - enforces limit <= 100
   - fills missing fields with defaults
5. Field validation is performed:
   - invalid fields trigger a refine prompt
6. The query is executed:
   - find/count/sum/top
7. Result formatting:
   - human-readable summary
   - details list with hierarchical structure

## Safety and UX Improvements
- Field projection: only requested fields are fetched when provided.
- Redaction: sensitive fields (password, token, secret, hash) are removed.
- No raw JSON in UI; details are a readable list.
- Clarification buttons when collection is unclear.
- Multi-step execution for stacked queries.

## Supported Query Types
- Count: "How many users do we have?"
- Find: "List movie titles and plots"
- Top: "Which movies have the highest rating?"
- Sum: supported for numeric fields

## Deployment Notes
- Render uses Docker to pin Python 3.11 and avoid Python 3.14 issues.
- runtime.txt exists but Docker is the most reliable approach.
- Keepalive is provided via GitHub Actions (or an external uptime service).
- Set env vars:
  - MONGODB_URI
  - LLM_API_KEY
  - LLM_MODEL

## Known Limitations
- Date-range aggregation and advanced filters are limited.
- The LLM is asked to keep filters simple.
- Field validation is based on a single sample document per collection.

## Files to Reference
- app/main.py: API endpoints
- app/db.py: MongoDB connection + schema snapshot
- app/agent.py: LLM planning, validation, execution, formatting
- static/index.html: UI layout
- static/styles.css: UI styling
- static/app.js: UI logic and details formatting
- Dockerfile: Render runtime
