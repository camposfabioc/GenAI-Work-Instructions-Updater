"""Thin OpenAI wrapper with automatic call logging for ops metrics."""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model constants — pinned once, used everywhere
# ---------------------------------------------------------------------------
CHEAP_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"

# Pricing per 1M tokens (USD), April 2026
_PRICING = {
    CHEAP_MODEL:  {"input": 0.15,  "output": 0.60},
    STRONG_MODEL: {"input": 2.50,  "output": 10.00},
}

SEED = 20260423

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Call log — append-only list, reset between runs
# ---------------------------------------------------------------------------
_call_log: list[dict] = []


def call_llm(
    messages: list[dict],
    model: str = CHEAP_MODEL,
    temperature: float = 0.0,
    response_format=None,
    **kwargs,
) -> "openai.types.chat.ChatCompletion":
    """Call OpenAI and log tokens + latency + cost.

    Parameters
    ----------
    messages : list[dict]
        Standard OpenAI messages array.
    model : str
        Model name — use CHEAP_MODEL or STRONG_MODEL constants.
    temperature : float
        Generation temperature.
    response_format : optional
        Structured output schema (Pydantic class or dict).
    **kwargs
        Forwarded to client.chat.completions.create / client.beta.chat.completions.parse.

    Returns
    -------
    ChatCompletion (or ParsedChatCompletion when response_format is a Pydantic class)
    """
    common = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=SEED,
        **kwargs,
    )

    t0 = time.time()

    if response_format is not None:
        resp = _client.beta.chat.completions.parse(
            response_format=response_format, **common
        )
    else:
        resp = _client.chat.completions.create(**common)

    latency = time.time() - t0

    usage = resp.usage
    pricing = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        usage.prompt_tokens * pricing["input"] / 1_000_000
        + usage.completion_tokens * pricing["output"] / 1_000_000
    )

    _call_log.append(
        {
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_s": round(latency, 3),
            "cost_usd": round(cost, 6),
        }
    )

    return resp


def get_call_log() -> list[dict]:
    """Return a copy of the call log."""
    return list(_call_log)


def reset_call_log() -> None:
    """Clear the call log. Call between pipeline runs."""
    _call_log.clear()
