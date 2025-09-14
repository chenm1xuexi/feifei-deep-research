# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based deep research application built with Chainlit for the frontend interface and LangGraph for workflow orchestration. The project is in early development stages with most implementation files currently containing placeholder content.

## Development Environment

The project uses:
- Python 3.12+ with uv package manager (uv.lock present)
- Chainlit for the web interface
- FastAPI for API services
- LangGraph for workflow management

## Common Development Commands

### Running the Application
```bash
# Run the Chainlit application directly
python chainlit_app/chainlit_api.py

# Or using Docker
docker-compose up --build
```

### Package Management
```bash
# Install dependencies using uv
uv sync

# Add new dependencies
uv add <package-name>
```

### Docker Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# The application will be available on port 8000
```

## Code Architecture

### Main Components

1. **Chainlit Interface** (`chainlit/`)
   - `chainlit_app.py`: Main Chainlit application entry point with basic chat handler setup
   - Uses LangGraph MemorySaver for conversation state management

2. **Deep Research Core** (`deep_research/`)
   - `workflow/`: Contains LangGraph workflow definitions (currently placeholder)
   - `agents/`: Agent implementations (currently placeholder) 
   - `llms/`: LLM integrations and configurations (currently placeholder)
   - `prompts/`: Prompt templates and management (currently placeholder)
   - `api/`: FastAPI application setup (currently placeholder)

3. **Common Utilities** (`common/`)
   - `log.py`: Logging utilities (currently empty)

4. **Frontend** (`frontend/`)
   - Pre-built static assets for the web interface
   - Contains various media player components and Chainlit UI assets

### Key Patterns

- The application follows a modular architecture with clear separation between UI (Chainlit), workflow logic (LangGraph), and API services (FastAPI)
- Session management is handled through Chainlit's user session system with thread-based conversation tracking
- The project structure suggests a multi-agent system design pattern

### Development Notes

- Most implementation files are currently empty or contain minimal placeholder code
- The main entry point is functional but lacks the core research logic implementation
- Docker configuration is present but commented out in the Dockerfile
- No test suite or linting configuration is currently present

## Project Status

This appears to be a newly initialized project with the basic structure in place but core functionality yet to be implemented. The README indicates this is a new version replacing an older implementation that was moved to the `old_deep_research` branch.