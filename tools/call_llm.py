"""
Utility for calling an OpenAI GPT‑4o chat model with optional tools,
ready to be plugged into LangChain / LangGraph graphs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


# Ensure variables from `.env` (including OPENAI_API_KEY) are loaded
load_dotenv()


def get_llm(model: str = "gpt-4o", temperature: float = 0.2) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.

    The OPENAI_API_KEY is expected to be available in the environment
    (e.g. via your .env file).
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
    )


def build_llm_runnable(
    tools: Optional[Sequence[BaseTool]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
) -> Runnable:
    """
    Build a Runnable that can be dropped into a LangGraph as an LLM node.

    - If `tools` are provided, the LLM is bound with those tools so it can
      call them using the OpenAI tool-calling / function-calling interface.
    - The resulting object supports `.invoke`, `.astream`, etc.
    """
    llm = get_llm(model=model, temperature=temperature)

    if tools:
        llm = llm.bind_tools(list(tools))

    def _runnable(input_data: Any) -> AIMessage:
        """
        Minimal runnable wrapper.

        Accepts either:
        - a plain string (treated as the user prompt), or
        - a list of LangChain messages.
        """
        if isinstance(input_data, str):
            messages = [HumanMessage(content=input_data)]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            raise TypeError(
                "build_llm_runnable expects a string or a list of messages "
                f"as input, got: {type(input_data)!r}"
            )

        return llm.invoke(messages)

    return Runnable(_runnable)  # type: ignore[arg-type]


def call_llm(
    prompt: str,
    tools: Optional[Sequence[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
) -> str:
    """
    Convenience helper for simple, one-off calls to the LLM.

    - `prompt`: user message
    - `tools`: optional list of LangChain tools the model is allowed to call
    - `system_prompt`: optional system message for behavior steering

    Returns the assistant's text content.
    """
    llm = get_llm(model=model, temperature=temperature)

    if tools:
        llm = llm.bind_tools(list(tools))

    messages: List[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    response: AIMessage = llm.invoke(messages)

    # For most use‑cases we just want the text content
    if isinstance(response.content, str):
        return response.content

    # Fallback: join any structured content into a string
    return " ".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in response.content  # type: ignore[union-attr]
    )

