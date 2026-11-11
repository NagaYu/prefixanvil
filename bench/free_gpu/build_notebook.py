#!/usr/bin/env python3
"""Build bench/free_gpu/prefixanvil_bench.ipynb with the PrefixAnvil source embedded.

The notebook is self-contained (no git clone, no Kaggle dataset upload): the
source needed on the GPU machine (src/, bench/, fixtures/{bench,equiv,tokenizer,
configs}, pyproject.toml, requirements.txt) is packed as base64 tar.gz into one
cell. Re-run this script after any code change and re-upload the notebook.

    .venv/bin/python bench/free_gpu/build_notebook.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "bench", "free_gpu", "prefixanvil_bench.ipynb")
INCLUDE = ["src/prefixanvil", "bench", "fixtures/bench", "fixtures/equiv", "fixtures/tokenizer",
           "fixtures/configs", "pyproject.toml", "requirements.txt", "README.md"]
EXCLUDE_PARTS = {"__pycache__", "results", ".pytest_cache"}


def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = set(ti.name.split("/"))
    if parts & EXCLUDE_PARTS or ti.name.endswith((".pyc", ".ipynb")):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0                                  # reproducible archive
    return ti


def pack() -> tuple[str, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for rel in INCLUDE:
            tf.add(os.path.join(ROOT, rel), arcname=f"prefixanvil/{rel}", filter=_filter)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()[:16]


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no", "--", ".",
                                ":!bench/free_gpu/prefixanvil_bench.ipynb"], capture_output=True, text=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    b64, digest = pack()
    rev = git_rev()
    cells = [
        md(f"""
# PrefixAnvil Phase 3 — free-tier GPU benchmark (Kaggle / Colab)

Runs the whole Step 1 measurement (and, on a 2-GPU Kaggle session, the time-boxed Step 2 Dynamo try)
in **one "Run all"**. Free tier only: no paid instance, no credit card, nothing is published.

* Kaggle: Settings → Accelerator **GPU T4 x2**, Internet **On**; then *Save Version → Save & Run All (Commit)*.
* Colab: Runtime → Change runtime type → **T4 GPU**; Runtime → Run all; keep the tab open.
* At the end, download the `prefixanvil_results_*.zip` (and `prefixanvil_dynamo_*.zip` if present).

Embedded source: git `{rev}`, archive sha256 `{digest}`. See `bench/free_gpu/RUNBOOK.md`.
"""),
        code("""
# --- settings (edit only if the RUNBOOK tells you to) ---
PROFILE = "full"          # "smoke" = ~30 min pipeline check with 40 requests per condition
RUN_ID = None             # set to a previous run id to resume it (see RUNBOOK)
RUN_DYNAMO = True         # Step 2 runs only when 2 GPUs are present and Step 1 finished
BUDGET_HOURS = 9.5        # Step 1 wall-clock budget (Kaggle sessions stop at 12 h)
EXTRA_ARGS = []           # e.g. ["--max-model-len", "6144", "--gpu-mem-util", "0.85"] after an out-of-memory start
RESUME_FROM = None        # Kaggle resume: path of a previous version's results dir under /kaggle/input/...

import os, sys, subprocess, shutil, json, datetime
PLATFORM = "kaggle" if os.path.isdir("/kaggle/working") else ("colab" if os.path.isdir("/content") else "other")
WORK = {"kaggle": "/kaggle/working/pf", "colab": "/content/pf"}.get(PLATFORM, os.path.abspath("pf"))
os.makedirs(WORK, exist_ok=True)
RUN_ID = RUN_ID or datetime.datetime.now(datetime.timezone.utc).strftime("t4-%Y%m%d-%H%M")
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, os.path.join(WORK, "results", RUN_ID), dirs_exist_ok=True)
    print("resuming from", RESUME_FROM)
print("platform:", PLATFORM, "| workdir:", WORK, "| run id:", RUN_ID)
smi = shutil.which("nvidia-smi")
if not smi:
    raise SystemExit("No GPU attached. Kaggle: Accelerator = GPU T4 x2. Colab: Runtime type = T4 GPU.")
q = subprocess.run([smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True).stdout.strip()
print(q)
N_GPUS = len([l for l in q.splitlines() if l.strip()])
print([l.strip() for l in subprocess.run([smi], capture_output=True, text=True).stdout.splitlines()
       if "CUDA Version" in l][:1])
"""),
        code(f"""
# --- unpack the embedded PrefixAnvil source (no network needed for this cell) ---
import base64, io, tarfile, hashlib
SRC_B64 = "{b64}"
raw = base64.b64decode(SRC_B64)
assert hashlib.sha256(raw).hexdigest()[:16] == "{digest}", "embedded source corrupted"
shutil.rmtree(os.path.join(WORK, "prefixanvil"), ignore_errors=True)
_tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
try:
    _tf.extractall(WORK, filter="data")
except TypeError:                     # Python < 3.11.4 has no extraction filters
    _tf.extractall(WORK)
SRC = os.path.join(WORK, "prefixanvil")
print("source unpacked to", SRC)
"""),
        code("""
# --- Step 1: install, model download, vLLM, all conditions, packaging (several hours) ---
def stream(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    return p.wait()

rc = stream([sys.executable, os.path.join(SRC, "bench/free_gpu/run_all.py"), "--src", SRC, "--workdir", WORK,
             "--run-id", RUN_ID, "--profile", PROFILE, "--budget-hours", str(BUDGET_HOURS), *EXTRA_ARGS])
print("run_all exit code:", rc)
"""),
        code("""
# --- Step 2: Dynamo 2-worker try (time-boxed 90 min; failure is an acceptable, recorded outcome) ---
if RUN_DYNAMO and N_GPUS >= 2:
    rc2 = stream([sys.executable, os.path.join(SRC, "bench/free_gpu/run_dynamo.py"), "--src", SRC,
                  "--workdir", WORK, "--run-id", RUN_ID, "--timebox-min", "90"])
    print("run_dynamo exit code:", rc2)
else:
    print(f"Step 2 skipped (RUN_DYNAMO={RUN_DYNAMO}, GPUs={N_GPUS})")
"""),
        code("""
# --- results: download these zip files and hand them back (see RUNBOOK) ---
zips = sorted(f for f in os.listdir(WORK) if f.endswith(".zip"))
for z in zips:
    print(os.path.join(WORK, z), os.path.getsize(os.path.join(WORK, z)) // 1024, "KiB")
if PLATFORM == "kaggle":
    from IPython.display import FileLink, display
    for z in zips:
        display(FileLink(os.path.relpath(os.path.join(WORK, z), "/kaggle/working")))
elif PLATFORM == "colab":
    from google.colab import files
    for z in zips:
        files.download(os.path.join(WORK, z))
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"},
                                         "language_info": {"name": "python"},
                                         "prefixanvil": {"git": rev, "archive_sha256": digest}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    nb = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KiB), source {nb['metadata']['prefixanvil']}")
