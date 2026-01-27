"""
LangGraph/LangChain-based tool orchestrator.

Given the tool the user wants to call (from the CLI menu in main.py),
we route through a LangGraph workflow and execute the appropriate agent.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents.email_generator import generate_email


class ToolState(TypedDict, total=False):
    """Shared state that flows through the LangGraph."""

    tool: str
    result: str


def _normalize_tool_choice(choice: str) -> str:
    """Normalize user choice into an internal tool identifier."""
    normalized = choice.strip().lower()

    if normalized in {"1", "resume analyzer", "1. resume_analyzer"}:
        return "resume_analyzer"
    if normalized in {"2", "email generator", "2. email_generator"}:
        return "email_generator"
    if normalized in {"3", "readme generator", "3. readme_generator"}:
        return "readme_generator"
    if normalized in {"4", "explain the codebase", "4. explain_codebase"}:
        return "explain_codebase"
    if normalized in {"5", "code review", "5. code_review"}:
        return "code_review"
    if normalized in {"6", "code summarizer", "6. code_summarizer"}:
        return "code_summarizer"
    if normalized in {"7", "web search", "7. web_search"}:
        return "web_search"
    if normalized in {"8", "job search", "8. job_search"}:
        return "job_search"
    if normalized in {"9", "chat with a llm", "9. chat_with_llm"}:
        return "chat_with_llm"
    if normalized in {"10", "exit", "10. exit"}:
        return "exit"

    return "unknown"


def _start_node(state: ToolState) -> ToolState:
    """Entry node that just returns the state."""
    return state


def _route_tool(state: ToolState) -> Literal[
    "resume_analyzer",
    "email_generator",
    "readme_generator",
    "explain_codebase",
    "code_review",
    "code_summarizer",
    "web_search",
    "job_search",
    "chat_with_llm",
    "exit",
    "unknown",
]:
    """Routing function used by LangGraph to pick the next node."""
    tool_id = _normalize_tool_choice(state.get("tool", ""))
    return tool_id  # type: ignore[return-value]


def _email_generator_node(state: ToolState) -> ToolState:
    """Node that runs the Email Generator agent."""
    print("Running Email Generator...")
    user_prompt = input(
        "Describe the email you want to write (purpose, tone, key points):\n> "
    )
    output_path = generate_email(user_prompt)
    result = f"Email generated and saved to: {output_path}"
    print(result)
    return {**state, "result": result}


def _exit_node(state: ToolState) -> ToolState:
    """Node that handles graceful exit."""
    print("Exiting...")
    return state


def _unknown_node(state: ToolState) -> ToolState:
    """Fallback node for unsupported / invalid choices."""
    print("Invalid choice. Please try again.")
    return state


def _build_workflow():
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(ToolState)

    # Nodes
    graph.add_node("start", _start_node)
    graph.add_node("email_generator", _email_generator_node)
    graph.add_node("exit", _exit_node)
    graph.add_node("unknown", _unknown_node)

    # Entry point
    graph.set_entry_point("start")

    # Conditional routing from the start node based on tool choice
    graph.add_conditional_edges(
        "start",
        _route_tool,
        {
            "email_generator": "email_generator",
            "exit": "exit",
            "unknown": "unknown",
        },
        # Any other tool id not listed above will cause the graph to end.
        # You can extend this map as you implement more tools.
    )

    # Terminal edges
    graph.add_edge("email_generator", END)
    graph.add_edge("exit", END)
    graph.add_edge("unknown", END)

    return graph.compile()


_APP = _build_workflow()


def orchestrate_tools(choice: str) -> None:
    """
    Public entrypoint used by main.py.

    Wraps the LangGraph workflow and passes the user's tool choice in the state.
    """
    state: ToolState = {"tool": choice}
    _APP.invoke(state)
