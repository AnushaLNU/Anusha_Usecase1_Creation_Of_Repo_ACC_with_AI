import os

from core.config import ANTHROPIC_MODEL, LLM_PROVIDER


def get_client():
    if LLM_PROVIDER != "anthropic":
        raise RuntimeError("Anthropic is not the configured LLM provider")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Anthropic API key is missing; set API_KEY in .env")

    try:
        from anthropic import Anthropic

        return Anthropic(api_key=api_key)
    except Exception as error:
        raise RuntimeError("Unable to initialize the Anthropic client") from error


def complete(prompt: str, max_tokens: int = 1200) -> str:
    try:
        response = get_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("Anthropic request failed") from error

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Anthropic response was truncated at the output token limit; "
            "increase max_tokens or reduce the prompt size"
        )

    content = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    if not content:
        raise RuntimeError("Anthropic returned an empty response")
    return content