"""Chat-template rendering and tokenization.

The tokenizer is pluggable through the :class:`ChatTokenizer` protocol. The
default implementation wraps a Hugging Face tokenizer loaded from a *local*
directory (``local_files_only=True``); downloads happen only when explicitly
allowed. Tests use the bundled fixture tokenizer under ``fixtures/tokenizer``.

Every render records where the chat template came from (bundled with the
tokenizer vs. an explicitly given file) and a content hash of the template, so
that template changes (F6) are attributable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from .ingest import Request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURE_TOKENIZER_DIR = os.path.join(REPO_ROOT, "fixtures", "tokenizer")


class ChatTokenizer(Protocol):
    name: str

    def template_info(self) -> dict[str, Any]: ...

    def render(self, messages: list[dict], tools: list[dict] | None,
               add_generation_prompt: bool) -> str: ...

    def encode(self, text: str) -> list[int]: ...


def template_version(template: str) -> str:
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


class HFChatTokenizer:
    """Hugging Face tokenizer + Jinja chat template (tokenizer only, no weights)."""

    def __init__(self, name_or_path: str, chat_template_path: str | None = None,
                 allow_download: bool = False):
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from transformers import AutoTokenizer  # local import: heavy

        self.name = name_or_path
        self._tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=not allow_download)
        if chat_template_path:
            with open(chat_template_path, encoding="utf-8") as f:
                self._template = f.read()
            self._source = f"explicit:{os.path.abspath(chat_template_path)}"
        else:
            tpl = getattr(self._tok, "chat_template", None)
            if isinstance(tpl, dict):
                tpl = tpl.get("tool_use") or tpl.get("default")
            if not tpl:
                raise ValueError(f"tokenizer {name_or_path!r} has no chat template; "
                                 "pass --chat-template explicitly")
            self._template = tpl
            self._source = f"tokenizer_bundled:{name_or_path}"

    def template_info(self) -> dict[str, Any]:
        return {
            "tokenizer": self.name,
            "tokenizer_is_fixture": os.path.abspath(self.name) == FIXTURE_TOKENIZER_DIR,
            "template_source": self._source,
            "template_version": template_version(self._template),
            # strftime_now makes the rendered prompt depend on wall-clock time:
            # an F1 source *inside the template*, invisible in request logs.
            "template_uses_strftime_now": "strftime_now" in self._template,
        }

    def render(self, messages: list[dict], tools: list[dict] | None,
               add_generation_prompt: bool) -> str:
        return self._tok.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=add_generation_prompt, chat_template=self._template)

    def encode(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=False)["input_ids"])


def load_tokenizer(name_or_path: str | None = None, chat_template_path: str | None = None,
                   allow_download: bool = False) -> HFChatTokenizer:
    return HFChatTokenizer(name_or_path or FIXTURE_TOKENIZER_DIR, chat_template_path, allow_download)


def flatten_messages(messages: list[dict]) -> list[dict]:
    """Return template-ready messages (content-part lists -> plain strings).

    Text parts are concatenated without separator; non-text parts become a
    ``<type>`` placeholder (multimodal hashing is out of Phase-1 scope).
    """
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            m = dict(m)
            m["content"] = "".join(p.get("text", "") if p.get("type") == "text"
                                   else f"<{p.get('type')}>" for p in c)
        out.append(m)
    return out


def vllm_normalize(messages: list[dict], tools: list[dict] | None) -> tuple[list[dict], list[dict] | None]:
    """Mimic what vLLM 0.33.0 does to a chat request *before* templating.

    Verified in the 0.33.0 source:
    * tools are re-serialized with ``ChatCompletionToolsParam.model_dump()``
      (renderers/online_renderer.py): top level ``{"type", "function"}``, function
      ``{"name", "description", "parameters"}`` in model-field order, missing
      description/parameters become ``null``; ``strict``/``defer_loading`` only
      when set. The content of ``parameters`` keeps the client's order.
    * assistant ``tool_calls[].function.arguments`` strings are parsed to dicts
      (renderers/chat_utils/chat_message.py ``_postprocess_messages``); invalid or
      non-object JSON becomes ``{}``; empty ``tool_calls`` lists are dropped.
    """
    out_tools = None
    if tools is not None:
        out_tools = []
        for t in tools:
            fn = t.get("function") or {}
            f = {"name": fn.get("name"), "description": fn.get("description"),
                 "parameters": fn.get("parameters")}
            for k in ("strict", "defer_loading"):
                if fn.get(k) is not None:
                    f[k] = fn[k]
            d = {"type": t.get("type", "function"), "function": f}
            if t.get("defer_loading") is not None:
                d["defer_loading"] = t["defer_loading"]
            out_tools.append(d)
    out_msgs = []
    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m:
            m = dict(m)
            if not m["tool_calls"]:
                m.pop("tool_calls")
            else:
                calls = []
                for tc in m["tool_calls"]:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    a = fn.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a else {}
                        except json.JSONDecodeError:
                            a = {}
                    if not isinstance(a, dict):
                        a = {}
                    fn["arguments"] = a
                    tc["function"] = fn
                    calls.append(tc)
                m["tool_calls"] = calls
        out_msgs.append(m)
    return out_msgs, out_tools


@dataclass
class Rendered:
    text: str
    token_ids: list[int]
    template_source: str
    template_version: str


class RenderError(RuntimeError):
    pass


class Renderer:
    """``server_normalization``: ``None`` renders the client's bytes (Phase 1/2
    behaviour); ``"vllm"`` applies :func:`vllm_normalize` first (Phase 3 fidelity)."""

    def __init__(self, tokenizer: ChatTokenizer, server_normalization: str | None = None):
        if server_normalization not in (None, "vllm"):
            raise ValueError("server_normalization must be None or 'vllm'")
        self.server_normalization = server_normalization
        self.tok = tokenizer
        info = tokenizer.template_info()
        self.template_source = info["template_source"]
        self.template_version = info["template_version"]

    def text(self, messages: list[dict], tools: list[dict] | None,
             add_generation_prompt: bool = True) -> str:
        try:
            messages = flatten_messages(messages)
            if self.server_normalization == "vllm":
                messages, tools = vllm_normalize(messages, tools)
            return self.tok.render(messages, tools, add_generation_prompt)
        except Exception as e:  # template errors are data errors, not crashes
            raise RenderError(f"{type(e).__name__}: {e}") from e

    def render(self, req: Request, add_generation_prompt: bool = True) -> Rendered:
        text = self.text(req.messages, req.tools, add_generation_prompt)
        return Rendered(text, self.tok.encode(text), self.template_source, self.template_version)
