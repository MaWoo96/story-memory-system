# Story Memory System

An AI interactive storytelling platform where persistent memory is the core differentiator. Users play through stories across multiple sessions (20, 50, 100+ sessions over months), and the AI remembers everything—characters, decisions, relationships, world facts, plot points, game stats—without manual maintenance.

## The Problem

Every AI storytelling app loses context. Sessions feel disconnected. The AI "forgets" what happened.

## The Solution

Automated memory extraction after each session, structured storage, and intelligent context retrieval.

## Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **Database**: Supabase (PostgreSQL)
- **Extraction LLM**: Grok 4.1 Fast (structured outputs)
- **Storytelling LLM**: Grok or Claude
- **Frontend**: Next.js (planned)

## Features

- **Automated Memory Extraction**: Extract entities, relationships, events, decisions after each session
- **Structured Storage**: Entity-centric database with temporal truth handling
- **Game State Tracking**: Stats, skills, inventory, NPC relationship meters
- **Context Building**: Intelligent retrieval of relevant memories for new sessions
- **Temporal Facts**: Track how information changes over time

## Project Structure

```
story-memory-system/
├── api/                    # FastAPI application
│   ├── main.py            # App entry point
│   ├── dependencies.py    # Dependency injection
│   └── routes/            # API endpoints
├── schemas/               # Pydantic models
│   ├── extraction.py     # Memory extraction schemas
│   └── api.py            # API request/response models
├── services/              # Business logic
│   ├── extraction.py     # LLM extraction service
│   ├── storage.py        # Database storage
│   └── context.py        # Context builder
├── db/                    # Database
│   ├── client.py         # Supabase client
│   └── migrations/       # SQL migrations
├── tests/                 # Tests
│   └── fixtures/         # Test transcripts
└── .env.example          # Environment variables template
```

## Setup

### 1. Clone and Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configure Secrets

**Development**: Use environment variables (`.env` file)
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials:
# - Supabase URL and keys (pre-configured)
# - xAI API key for Grok 4.1 Fast (get at https://console.x.ai)
# - Optional: Anthropic API key for Claude
```

**Production**: Use AWS Secrets Manager or Google Cloud Secret Manager

See **[Secret Management Guide](docs/SECRET_MANAGEMENT.md)** for detailed setup instructions for all environments.

### 3. Set Up Database

**Database is already set up!** The Supabase project is configured and all migrations have been applied:
- Project URL: `https://mntpiewbprdjpgcbzaca.supabase.co`
- 17 tables created with Row Level Security (RLS) enabled
- All security policies configured

No manual SQL setup required.

### 4. Run the API

```bash
# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for the interactive API documentation.

## API Endpoints

### Session Management
- `POST /api/sessions/{story_id}/complete` - Process completed session
- `GET /api/sessions/{story_id}/context` - Get context for new session

### Memory Queries
- `GET /api/stories/{story_id}/summary` - Get story summary
- `GET /api/stories/{story_id}/state` - Get game state (stats, inventory, NPCs)
- `GET /api/stories/{story_id}/search` - Search story memories

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Testing Grok 4.1 Fast Extraction

Quick test with the included test script:

```bash
python test_grok_extraction.py
```

Or use the extraction service directly:

```python
from services.extraction import ExtractionService

service = ExtractionService()
with open('tests/fixtures/sample_transcript.txt') as f:
    result = service.extract_session(f.read())
print(result.model_dump_json(indent=2))
```

## Architecture

### Memory Extraction Pipeline

1. **Session Transcript** → User interacts with storytelling LLM
2. **Extraction** → Grok 4.1 Fast + Pydantic schemas → Structured JSON
3. **Storage** → Entities, facts, relationships, events stored in PostgreSQL
4. **Retrieval** → Context builder queries relevant memories for new session

### What Gets Extracted

- **Entities**: Characters, locations, items, factions, concepts
- **Relationships**: Connections between entities with types
- **Events**: Plot points, revelations, conflicts
- **Decisions**: Player choices (made or pending)
- **Protagonist State**: Stats, skills, inventory, status effects
- **Character States**: NPC affection/trust/loyalty meters
- **World State**: Current time, location, obligations

## Database Schema

See `db/migrations/` for the complete PostgreSQL schema including:

- Entity-centric storage with aliases
- Temporal truth tracking (facts can be invalidated)
- Relationship mapping
- Event timeline
- Game mechanics (stats, skills, inventory)
- NPC relationship meters

## Implementation Status

- [x] Project setup
- [x] Database schema (Supabase PostgreSQL)
- [x] Row Level Security (RLS) policies
- [x] Pydantic extraction schemas
- [x] Grok 4.1 Fast extraction service (OpenAI-compatible)
- [ ] Storage service
- [ ] Context builder
- [ ] API endpoints
- [ ] Frontend (Next.js)

## Grok 4.1 Fast Integration

This project uses **Grok 4.1 Fast** (`grok-4.1-fast-reasoning`) from xAI for structured memory extraction:

- **Model**: Grok 4.1 Fast with reasoning (2M context window)
- **API**: OpenAI-compatible via `https://api.x.ai/v1`
- **Features**: Structured outputs with Pydantic schemas, tool calling, agent capabilities
- **Cost**: Free agent tools API

The extraction service uses OpenAI's structured output API with Pydantic models to guarantee schema compliance.

## Security & Secret Management

The project includes a comprehensive secret management system that supports multiple backends:

- **Development**: Environment variables (`.env` files)
- **Production**: AWS Secrets Manager or Google Cloud Secret Manager

All secrets (API keys, database credentials) are centralized through the `SecretManager` class, making it easy to switch between environments without code changes.

**Key Features:**
- ✅ Secrets never committed to version control
- ✅ In-memory caching for performance
- ✅ Support for AWS and GCP secret rotation
- ✅ IAM-based access control
- ✅ Easy migration between backends

**Learn more**: See [Secret Management Guide](docs/SECRET_MANAGEMENT.md)

## Documentation

- [Secret Management Guide](docs/SECRET_MANAGEMENT.md) - Comprehensive guide for managing secrets in dev and production

## License

MIT
