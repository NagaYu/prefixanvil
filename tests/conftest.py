import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"          # tests must never touch the network
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def renderer():
    from prefixanvil.render import Renderer, load_tokenizer
    return Renderer(load_tokenizer())


@pytest.fixture(scope="session")
def root():
    return ROOT
