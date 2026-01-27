"""
Email Generator agent.

Takes a single user prompt describing the email the user wants to write,
calls the LLM to draft the email, and stores the output in a text file.

Outputs are stored under:
    output/tool_two/email_one.txt
    output/tool_two/email_two.txt
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from tools.call_llm import call_llm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "tool_two"


_NUMBER_WORDS: Dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _index_to_word(index: int) -> str:
    """
    Convert an index (1‑based) to a word suffix.

    For indices beyond the predefined mapping, fall back to the number itself.
    """
    return _NUMBER_WORDS.get(index, str(index))


def _next_output_path() -> Path:
    """
    Determine the next email file path inside output/tool_two.

    The pattern used is: email_one.txt, email_two.txt, etc.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in OUTPUT_DIR.glob("email_*.txt") if p.is_file()
    )
    next_index = len(existing) + 1
    suffix = _index_to_word(next_index)
    return OUTPUT_DIR / f"email_{suffix}.txt"


def generate_email(user_prompt: str) -> Path:
    """
    Generate an email using the LLM and persist it to a new file.

    Returns the path to the created file.
    """
    system_prompt = (
        "You are an expert email writer. "
        "Given a short description of the email a user wants to send, "
        "write a clear, concise, and well‑formatted email. "
        "Do not include any explanations, only the final email body."
    )

    email_text = call_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    output_path = _next_output_path()
    output_path.write_text(email_text, encoding="utf-8")
    return output_path


def run() -> None:
    """
    CLI entrypoint for the Email Generator agent.
    """
    user_prompt = input(
        "Describe the email you want to write (purpose, tone, key points):\n> "
    )

    output_path = generate_email(user_prompt)
    print(f"Email generated and saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()

