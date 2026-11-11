"""Build the small fixture tokenizer used by tests (offline, deterministic).

This is NOT a real model tokenizer. It is a byte-level BPE with a 2048-token
vocabulary trained on AgentLoad text, so that token boundaries behave like a
real BPE (merges across spaces, JSON punctuation, etc.) while staying tiny.

Regenerate (only needed if the template or corpus changes):

    .venv/bin/python fixtures/tokenizer/build_fixture_tokenizer.py

The outputs (tokenizer.json, tokenizer_config.json) are committed; tests never
run this script.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from prefixanvil.agentload import FAULTS, AgentLoadConfig, generate  # noqa: E402

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def corpus() -> list[str]:
    cfg = AgentLoadConfig.from_dict({"seed": 12345, "sessions": 30, "turns": 2,
                                     "faults": {f: 0.1 for f in FAULTS}})
    reqs, _ = generate(cfg)
    texts = []
    for r in reqs:
        for m in r["messages"]:
            if m.get("content"):
                texts.append(m["content"])
            for tc in m.get("tool_calls") or []:
                texts.append(json.dumps(tc))
        for t in r.get("tools") or []:
            texts.append(json.dumps(t))
    texts.append("2026-11-05T10:22:31Z 550e8400-e29b-41d4-a716-446655440000 run-000123")
    return texts


def main() -> None:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=2048, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(corpus(), trainer=trainer)
    tok.save(os.path.join(HERE, "tokenizer.json"))

    with open(os.path.join(HERE, "chat_template.src.jinja"), encoding="utf-8") as f:
        template = f.read()
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 131072,
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "add_bos_token": False,
        "clean_up_tokenization_spaces": False,
        "chat_template": template,
    }
    with open(os.path.join(HERE, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("vocab size:", tok.get_vocab_size())


if __name__ == "__main__":
    main()
