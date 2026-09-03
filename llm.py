"""LLM provider layer for the chatbot.

The agent loop in :mod:`chatbot` is written against the small interface below,
so the same loop runs on either backend:

* **groq** (default) - Groq's OpenAI-compatible chat completions API.
* **anthropic** - the Anthropic SDK.

Which one is used is decided by ``LLM_PROVIDER``, or inferred from whichever
API key is present. Tools are declared once, in Anthropic's schema shape
(see :data:`agent_tools.TOOLS`), and converted for Groq here - so adding a tool
means touching one file, not two.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

MAX_TOKENS = 8000


@dataclass
class ToolCall:
    """One tool invocation, normalised across providers."""
    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """One assistant turn."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(RuntimeError):
    """Configuration or API failure worth showing the user verbatim."""


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class GroqProvider:
    name = "groq"
    label = "Groq"

    def __init__(self, api_key: str, model: str | None = None):
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The `groq` package is not installed. Run: pip install groq"
            ) from exc
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        # Groq's free tier caps tokens-per-minute; let the SDK back off and
        # retry a 429 rather than surfacing it as a failed answer mid-demo.
        self._client = Groq(api_key=api_key, max_retries=5, timeout=120.0)

    @staticmethod
    def convert_tools(tools: list[dict]) -> list[dict]:
        """Anthropic tool schema -> OpenAI/Groq function schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def start(self, system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}]

    def call(self, messages: list[dict], tools: list[dict],
             system_prompt: str = "") -> Turn:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.convert_tools(tools),
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=0.0,  # deterministic SQL matters more than variety here
        )
        message = response.choices[0].message

        calls = []
        for call in message.tool_calls or []:
            # Arguments arrive as a JSON string; a malformed one is the model's
            # problem to fix, so surface it as an empty input and let the tool
            # layer return a correctable error.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__parse_error__": call.function.arguments}
            calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return Turn(text=message.content or "", tool_calls=calls, raw=message)

    def append_assistant(self, messages: list[dict], turn: Turn) -> None:
        entry: dict = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in turn.tool_calls
            ]
        messages.append(entry)

    def append_tool_results(self, messages: list[dict], results: list[tuple]) -> None:
        # OpenAI-shaped APIs take one message per tool result.
        for call, content, _is_error in results:
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"
    label = "Claude"

    def __init__(self, api_key: str | None, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The `anthropic` package is not installed. Run: pip install anthropic"
            ) from exc
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        # A bare client also resolves an `ant auth login` profile, so only pass
        # the key when we actually have one.
        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def start(self, system_prompt: str) -> list[dict]:
        # Anthropic carries the system prompt outside the message list, so the
        # history starts empty.
        return []

    def call(self, messages: list[dict], tools: list[dict],
             system_prompt: str = "") -> Turn:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return Turn(text=text, tool_calls=calls, raw=response)

    def append_assistant(self, messages: list[dict], turn: Turn) -> None:
        messages.append({"role": "assistant", "content": turn.raw.content})

    def append_tool_results(self, messages: list[dict], results: list[tuple]) -> None:
        # All results for one assistant turn go back in a single user message.
        blocks = []
        for call, content, is_error in results:
            block = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        messages.append({"role": "user", "content": blocks})


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def resolve_provider():
    """Pick a provider from the environment.

    Returns the provider, or raises ProviderError with a message the UI can show.
    """
    requested = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if requested == "groq" or (not requested and groq_key):
        if not groq_key:
            raise ProviderError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. Add it to your .env file."
            )
        return GroqProvider(groq_key)

    if requested == "anthropic" or (not requested and anthropic_key):
        return AnthropicProvider(anthropic_key)

    raise ProviderError(
        "No API key found. Set GROQ_API_KEY (or ANTHROPIC_API_KEY) in a .env "
        "file next to app.py, then restart the app."
    )
