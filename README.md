# Toolhub

Toolhub is a cli-based AI assistant that uses LangChain, LangGraph, and OpenAI GPT‑4o to orchestrate multiple specialized agents (resume analyzer, email generator, code tools, web search, job search, and more) from a single menu-driven interface.

## Features

- **Central CLI menu**: Pick from multiple agents in one place.
- **Email generator**: Generates well‑structured emails and saves them under `output/tool_two/` as versioned `.txt` files.
- **Extensible design**: New agents can be added easily and wired into the LangGraph workflow.

### Requirements

- Python 3.13+
- `uv` package manager
- OpenAI API key in an `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

### Setup

From the project root:

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` / `uv.lock`.

### Running the CLI

From the project root:

```bash
uv run python -m main
```

Then follow the on-screen menu to pick a tool (for example, the Email Generator).

### Project structure (high level)

- `main.py` – CLI entrypoint that shows the menu and forwards the choice to the orchestrator.
- `orchestrator.py` – LangGraph-based workflow that routes the selected tool to the correct agent.
- `agents/` – Individual agents (e.g. `email_generator.py`).
- `tools/` – Shared utilities (e.g. `call_llm.py` for LLM access).
- `output/` – Generated artifacts from agents (e.g. generated emails).

### Notes

- The project is designed so that each agent encapsulates its own logic and IO, while sharing a common LLM interface and orchestration graph.
