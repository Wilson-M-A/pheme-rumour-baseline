"""
Competing-Wisdom explanation generation for the PHEME project.

WHY ENGLISH: the downstream classifier is bert-base-uncased (English-only vocab).
Chinese explanations would tokenise to mostly [UNK] and be silently discarded,
making any "no improvement" result meaningless. Claims are English; the arguments
must quote their actual wording ("BREAKING", "Reports of"), so English throughout.

Findings so far (measured, not assumed):
  - thinking is a waste here: 19k chars of draft for 3 sentences.
      local Qwen3.5-4B, thinking ON : 141 s/call -> ~504 h full run. Infeasible.
      local Qwen3.5-4B, thinking OFF:   5 s/call -> ~18 h full run. Feasible but slow.
  - thinking OFF exposed two failure modes, both fixed below by prompt design:
      (1) stance collapse: the NOT-A-RUMOUR side arguing "actually this is a rumour"
      (2) hallucinated outside facts: importing world knowledge absent from the post
  - MiniMax M3 + concurrency: ~1 h, ~$6.

MODEL CHOICE: must be MiniMax-M3. The M2.x series cannot disable thinking --
it accepts the parameter but still generates (and bills for) thousands of tokens.

Usage:
  uv add openai ollama python-dotenv
  # .env  ->  MINIMAX_API_KEY=xxx     (make sure .env is gitignored)
  uv run python competing_wisdom.py
"""

import os
import re
import time
import ollama
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()   # must run before reading os.environ

# ═══ Config ═══
BACKEND = "minimax"        # "minimax" | "ollama"
THINK = False              # settled by the timing experiment above

OLLAMA_MODEL = "qwen3.5:4b"
MINIMAX_MODEL = "MiniMax-M3"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MAX_WORKERS = 8            # start here; lower it if you hit 429s

RUMOUR = "A RUMOUR"
NOT_RUMOUR = "NOT A RUMOUR"

# ═══ Prompt (identical for both sides -- symmetry is the whole point) ═══
SYSTEM_PROMPT = """You are a defense lawyer. Your client is a GIVEN VERDICT, and your only job is to write the strongest possible defense of it.
You are not the judge. You do not decide what is true — someone else does that.

You may argue only from clues observable in the post's own text:
word choice, whether the claim can be checked, whether a source is named, internal logical consistency, and any emotional exaggeration or manipulation.

STRICT PROHIBITIONS — violating any of these makes your output invalid:
1. Support the given verdict from the first word to the last. Never use "however", "but", "nevertheless", "that said", "on the other hand", "in reality", "if we were to classify it as", or any phrase that pivots away from, questions, or walks back the given verdict. Never write "this looks like X, but if we call it Y then...".
2. Never bring in factual knowledge from outside the post. Do not mention any background knowledge you have about the people, organisations, places, or events named in it. You know nothing about the world; you can only see this piece of text.
3. Even if the given verdict seems obviously wrong to you, argue for it with full force. That is your professional duty.

Format: 3-4 sentences of English prose. Concise, specific, grounded in the text. Output the defense directly, with no preamble."""

FEWSHOT_USER = """Post: "Sources say the central bank governor of a certain country resigned last night; officials have not yet responded."
Given verdict: this post is NOT A RUMOUR.
Argue why it may be NOT A RUMOUR."""

FEWSHOT_ASSISTANT = """The post attributes its claim to "sources say", presenting the information as secondhand reporting rather than asserting it as established fact, which reflects careful handling of an uncertain claim. It volunteers that "officials have not yet responded", openly conceding that the account remains unconfirmed — a self-limiting move that contrasts with fabrications engineered to look settled. The text stays on a single concrete event and adds no inflammatory commentary or emotive cues, keeping its scope narrow enough to be checked against an official statement."""

USER_TEMPLATE = """Post: "{claim}"
Given verdict: this post is {label}.
Argue why it may be {label}."""


def build_messages(claim: str, label: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": USER_TEMPLATE.format(claim=claim, label=label)},
    ]

# ═══ QC: catch the two failure modes automatically ═══
# High-precision reversal markers. Bare "but" is excluded on purpose --
# "not X but Y" is legitimate, so it would false-positive constantly.
STANCE_BREAK = [
    r"\bhowever\b", r"\bnevertheless\b", r"\bnonetheless\b", r"\bconversely\b",
    r"\badmittedly\b", r"\bthat said\b", r"\bon the other hand\b",
    r"\bin reality\b", r"\bin truth\b", r"\bif we were to\b", r"\bif one were to\b",
]
META_TALK = [r"\bas an ai\b", r"\bi cannot\b", r"\bi can't\b", r"\bit is worth noting\b"]

def qc_check(text: str):
    issues = []
    low = text.lower()
    hits = [p.strip("\\b") for p in STANCE_BREAK if re.search(p, low)]
    if hits:
        issues.append(f"possible stance collapse ({'/'.join(hits)})")
    if any(re.search(p, low) for p in META_TALK):
        issues.append("meta-commentary / refusal")
    if len(text) < 80:
        issues.append("too short")
    return issues

# ═══ Backends ═══
_client = None

def _get_minimax_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise RuntimeError("MINIMAX_API_KEY not found. Put it in .env (and gitignore .env).")
        _client = OpenAI(api_key=key, base_url=MINIMAX_BASE_URL)
    return _client


def generate_argument(claim: str, label: str):
    """Returns (argument_text, seconds)."""
    t0 = time.perf_counter()
    msgs = build_messages(claim, label)

    if BACKEND == "ollama":
        resp = ollama.chat(
            model=OLLAMA_MODEL, messages=msgs, think=THINK,
            options={"temperature": 0.4, "seed": 42, "num_ctx": 4096},
        )
        text = resp.message.content.strip()

    elif BACKEND == "minimax":
        client = _get_minimax_client()
        kwargs = {"model": MINIMAX_MODEL, "messages": msgs,
                  "temperature": 0.4, "max_tokens": 400}
        if not THINK:
            # M3 only. M2.x accepts this but keeps thinking (and billing for it).
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
    else:
        raise ValueError(f"unknown BACKEND: {BACKEND}")

    return text, time.perf_counter() - t0


def generate_competing(claim: str):
    r_arg, r_t = generate_argument(claim, RUMOUR)
    n_arg, n_t = generate_argument(claim, NOT_RUMOUR)
    return {
        "claim": claim,
        "rumour_argument": r_arg,
        "nonrumour_argument": n_arg,
        "seconds": r_t + n_t,
        "qc_rumour": qc_check(r_arg),
        "qc_nonrumour": qc_check(n_arg),
    }


def generate_batch(claims, max_workers=MAX_WORKERS):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(generate_competing, c): c for c in claims}
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            try:
                results[c] = fut.result()
            except Exception as e:
                print(f"  x failed: {c[:40]}... -> {e}")
            if i % 20 == 0:
                print(f"  progress {i}/{len(claims)}")
    return [results[c] for c in claims if c in results]


if __name__ == "__main__":
    # Both samples sit in the event-based TEST set (ottawashooting + putinmissing)
    # and represent the two failure directions found in the error analysis:
    # FP concentrated in ottawashooting, FN concentrated in putinmissing.
    samples = [
        "BREAKING: Reports of gunfire at Parliament Hill in Ottawa, multiple shots heard, area on lockdown.",
        "Putin has secretly gone missing for 10 days and a coup is underway inside the Kremlin.",
    ]

    model = MINIMAX_MODEL if BACKEND == "minimax" else OLLAMA_MODEL
    print(f"backend={BACKEND} | model={model} | thinking={THINK}\n")

    t0 = time.perf_counter()
    outs = generate_batch(samples, max_workers=min(MAX_WORKERS, len(samples)))
    wall = time.perf_counter() - t0

    for out in outs:
        print("=" * 72)
        print("CLAIM:", out["claim"])
        print(f"\n[RUMOUR side]  {'! ' + '; '.join(out['qc_rumour']) if out['qc_rumour'] else 'QC ok'}")
        print(" " + out["rumour_argument"])
        print(f"\n[NOT-A-RUMOUR side]  {'! ' + '; '.join(out['qc_nonrumour']) if out['qc_nonrumour'] else 'QC ok'}")
        print(" " + out["nonrumour_argument"])
        print(f"\n--> {out['seconds']:.1f}s of call time")

    N_CLAIMS = 6406          # cleaned PHEME size
    n_calls = len(samples) * 2
    avg = sum(o["seconds"] for o in outs) / n_calls
    print("\n" + "=" * 72)
    print(f"wall {wall:.1f}s | avg {avg:.1f}s per call")
    print(f"extrapolated: {N_CLAIMS} claims x 2 sides at {MAX_WORKERS} workers "
          f"-> ~{avg * N_CLAIMS * 2 / MAX_WORKERS / 3600:.1f} h")