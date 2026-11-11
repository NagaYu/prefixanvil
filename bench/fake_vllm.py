"""Fake vLLM OpenAI server for offline harness tests (no GPU, no model).

Implements the slice of vLLM 0.33.0 the harness uses: streaming
``/v1/chat/completions`` with usage + ``prompt_tokens_details.cached_tokens``,
Prometheus ``/metrics`` (prefix-cache counters, prefill-KV histogram,
``vllm:cache_config_info``), ``/health`` and the dev-mode
``POST /reset_prefix_cache``. Prefix caching is the PrefixAnvil block simulator
over the fixture tokenizer, rendered with vLLM-style request normalization, so
fidelity against it must be exact — which is what the tests assert.

``name_style``: ``"v033"`` (current names), ``"legacy"`` (``vllm:gpu_prefix_cache_*``),
``"missing"`` (no prefix-cache counters at all).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prefixanvil.ingest import Request
from prefixanvil.mock_upstream import mock_completion_text
from prefixanvil.render import Renderer, load_tokenizer
from prefixanvil.simulator import PrefixCacheSimulator, SimConfig


class FakeVLLM:
    def __init__(self, block_size: int = 16, num_blocks: int = 4096, dev_mode: bool = True,
                 name_style: str = "v033", served_model: str = "pf-model"):
        self.block_size, self.num_blocks = block_size, num_blocks
        self.dev_mode, self.name_style, self.served_model = dev_mode, name_style, served_model
        self.renderer = Renderer(load_tokenizer(), "vllm")
        self.sim = PrefixCacheSimulator(SimConfig(block_size, num_blocks))
        self.c = {"hits": 0, "queries": 0, "prompt": 0, "cached": 0, "gen": 0, "kv_sum": 0, "kv_count": 0, "ok": 0}
        self.lock = threading.Lock()
        self.resets = 0
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._srv.daemon_threads = True
        self._srv.fake = self

    @property
    def url(self) -> str:
        h, p = self._srv.server_address[:2]
        return f"http://{h}:{p}"

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()

    def metrics(self) -> str:
        c, L = self.c, []
        lab = f'engine="0",model_name="{self.served_model}"'
        if self.name_style != "missing":
            pre = "vllm:gpu_prefix_cache" if self.name_style == "legacy" else "vllm:prefix_cache"
            L += [f"# TYPE {pre}_queries_total counter", f"{pre}_queries_total{{{lab}}} {float(c['queries'])}",
                  f"{pre}_queries_created{{{lab}}} 1.7e9",
                  f"# TYPE {pre}_hits_total counter", f"{pre}_hits_total{{{lab}}} {float(c['hits'])}"]
        L += [f"vllm:prompt_tokens_total{{{lab}}} {float(c['prompt'])}",
              f"vllm:prompt_tokens_cached_total{{{lab}}} {float(c['cached'])}",
              f"vllm:generation_tokens_total{{{lab}}} {float(c['gen'])}",
              f'vllm:request_success_total{{{lab},finished_reason="length"}} {float(c["ok"])}',
              f"vllm:request_prefill_kv_computed_tokens_sum{{{lab}}} {float(c['kv_sum'])}",
              f"vllm:request_prefill_kv_computed_tokens_count{{{lab}}} {float(c['kv_count'])}",
              f'vllm:cache_config_info{{block_size="{self.block_size}",enable_prefix_caching="True",'
              f'engine="0",num_gpu_blocks="{self.num_blocks}",prefix_caching_hash_algo="sha256"}} 1.0']
        return "\n".join(L) + "\n"

    def complete(self, body: dict) -> tuple[str, dict]:
        req = Request(0, 0, body.get("model", ""), body["messages"], body.get("tools") or None, None, None, None)
        toks = self.renderer.render(req).token_ids
        words = mock_completion_text(body).split(" ")[: int(body.get("max_tokens") or 16)]
        text = " ".join(words)
        with self.lock:
            r = self.sim.process(toks)
            c = self.c
            c["queries"] += r.prompt_tokens
            c["hits"] += r.cached_tokens
            c["prompt"] += r.prompt_tokens
            c["cached"] += r.cached_tokens
            c["gen"] += len(words)
            c["kv_sum"] += r.prefill_tokens
            c["kv_count"] += 1
            c["ok"] += 1
        usage = {"prompt_tokens": r.prompt_tokens, "completion_tokens": len(words),
                 "total_tokens": r.prompt_tokens + len(words),
                 "prompt_tokens_details": {"cached_tokens": r.cached_tokens}}
        return text, usage

    def reset(self) -> None:
        with self.lock:
            self.sim = PrefixCacheSimulator(SimConfig(self.block_size, self.num_blocks))
            self.resets += 1


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status: int, data: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        fake: FakeVLLM = self.server.fake
        if self.path == "/health":
            return self._send(200, b"")
        if self.path == "/metrics":
            return self._send(200, fake.metrics().encode(), "text/plain; version=0.0.4")
        self._send(404, b'{"detail":"Not Found"}')

    def do_POST(self):
        fake: FakeVLLM = self.server.fake
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        path = self.path.split("?")[0]
        if path == "/reset_prefix_cache":
            if not fake.dev_mode:
                return self._send(404, b'{"detail":"Not Found"}')
            fake.reset()
            return self._send(200, b'{"success": true}')
        if path != "/v1/chat/completions":
            return self._send(404, b'{"detail":"Not Found"}')
        body = json.loads(raw)
        text, usage = fake.complete(body)
        if not body.get("stream"):
            return self._send(200, json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
                                               "usage": usage}).encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(obj):
            data = f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
        for i, w in enumerate(text.split(" ")):
            chunk({"choices": [{"index": 0, "delta": {"content": w if i == 0 else " " + w}}]})
        chunk({"choices": [], "usage": usage})
        chunk("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
