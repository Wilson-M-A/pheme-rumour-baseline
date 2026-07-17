"""
Full-run competing-wisdom explanation generation over pheme_clean.csv.

DESIGN DECISIONS (recorded here so they end up in the README):
  - Input column is `text`, NOT `text_raw`. The baseline BERT was trained on `text`,
    so feeding `text` + two arguments keeps the ONLY variable = the arguments.
    Using `text_raw` would restore the @usernames and URLs that `text` anonymised,
    confounding "did competing wisdom help?" with "did leaking sources back help?".
  - The LLM NEVER sees the gold label. It argues both sides blind. Any label
    reaching the generator would be leakage and would invalidate the whole run.
  - Resumable: results are appended to JSONL as they complete, and an existing
    file is read on startup so finished claims are skipped. A 4-hour run WILL be
    interrupted (rate limits, network, Ctrl-C); restarting from zero is not an option.

Usage:
  uv run python generate_explanations.py
  # safe to Ctrl-C and re-run; it picks up where it stopped
"""

import os
import re
import csv
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ═══ Config ═══
IN_CSV = "data/pheme_clean.csv"                       # adjust if yours lives elsewhere
OUT_DIR = "data/explanations"
OUT_JSONL = os.path.join(OUT_DIR, "competing_wisdom_minimax_m3.jsonl")

MODEL = "MiniMax-M3"                             # M2.x cannot disable thinking -> do not use
BASE_URL = "https://api.minimaxi.com/v1"         # CN site; use api.minimax.io for international
MAX_WORKERS = 4                                  # Token Plan keys rate-limit hard; pay-as-you-go can go higher
TEMPERATURE = 0.4
MAX_TOKENS = 400

# MiniMax M3 pricing, USD per 1M tokens (promotional tier)
PRICE_IN, PRICE_OUT = 0.30, 1.20

RUMOUR = "A RUMOUR"
NOT_RUMOUR = "NOT A RUMOUR"

# ═══ Prompt (validated on ottawashooting + putinmissing samples) ═══
SYSTEM_PROMPT = """You are a defense lawyer. Your client is a GIVEN VERDICT, and your only job is to write the strongest possible defense of it.
You are not the judge. You do not decide what is true — someone else does that.

You may argue only from clues observable in the post's own text:
word choice, whether the claim can be checked, whether a source is named, internal logical consistency, and any emotional exaggeration or manipulation.

STRICT PROHIBITIONS — violating any of these makes your output invalid:
1. Support the given verdict from the first word to the last. Never use "however", "but", "nevertheless", "that said", "on the other hand", "in reality", "if we were to classify it as", or any phrase that pivots away from, questions, or walks back the given verdict. Never write "this looks like X, but if we call it Y then...".
2. Never bring in factual knowledge from outside the post. Do not mention any background knowledge you have about the people, organisations, places, or events named in it. You know nothing about the world; you can only see this piece of text.
3. Even if the given verdict seems obviously wrong to you, argue for it with full force. That is your professional duty.

Note: the posts are tweets in which usernames have been replaced by @USER and links by [URL]. Treat these as anonymised placeholders — you can observe THAT a user or link was referenced, but not which one.

Format: 3-4 sentences of English prose. Concise, specific, grounded in the text. Output the defense directly, with no preamble."""

FEWSHOT_USER = """Post: "Sources say the central bank governor of a certain country resigned last night; officials have not yet responded."
Given verdict: this post is NOT A RUMOUR.
Argue why it may be NOT A RUMOUR."""

FEWSHOT_ASSISTANT = """The post attributes its claim to "sources say", presenting the information as secondhand reporting rather than asserting it as established fact, which reflects careful handling of an uncertain claim. It volunteers that "officials have not yet responded", openly conceding that the account remains unconfirmed — a self-limiting move that contrasts with fabrications engineered to look settled. The text stays on a single concrete event and adds no inflammatory commentary or emotive cues, keeping its scope narrow enough to be checked against an official statement."""

USER_TEMPLATE = """Post: "{claim}"
Given verdict: this post is {label}.
Argue why it may be {label}."""


def build_messages(claim, label):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": USER_TEMPLATE.format(claim=claim, label=label)},
    ]

# ═══ QC ═══
STANCE_BREAK = [r"\bhowever\b", r"\bnevertheless\b", r"\bnonetheless\b", r"\bconversely\b",
                r"\badmittedly\b", r"\bthat said\b", r"\bon the other hand\b",
                r"\bin reality\b", r"\bin truth\b", r"\bif we were to\b", r"\bif one were to\b"]
META_TALK = [r"\bas an ai\b", r"\bi cannot\b", r"\bi can't\b", r"\bit is worth noting\b"]

def qc_check(text):
    issues = []
    low = text.lower()
    hits = [p.strip("\\b") for p in STANCE_BREAK if re.search(p, low)]
    if hits:
        issues.append("stance_break:" + "/".join(hits))
    if any(re.search(p, low) for p in META_TALK):
        issues.append("meta_talk")
    if len(text) < 80:
        issues.append("too_short")
    return issues

# ═══ Client & counters ═══
client = OpenAI(api_key=os.environ["MINIMAX_API_KEY"], base_url=BASE_URL)

_lock = threading.Lock()
_usage = {"in": 0, "out": 0, "calls": 0}


MAX_RETRIES = 6          # 429s are usually transient: back off and try again
BASE_BACKOFF = 4.0       # seconds; doubles each retry -> 4, 8, 16, 32, 64, 128


def generate_argument(claim, label):
    from openai import RateLimitError, APIError

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(claim, label),
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"thinking": {"type": "disabled"}},   # M3 only; omit -> thinking ON
            )
            u = resp.usage
            with _lock:
                _usage["in"] += u.prompt_tokens
                _usage["out"] += u.completion_tokens
                _usage["calls"] += 1
            return (resp.choices[0].message.content or "").strip()

        except (RateLimitError, APIError) as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            wait = BASE_BACKOFF * (2 ** attempt)
            with _lock:
                _usage["retries"] = _usage.get("retries", 0) + 1
            time.sleep(wait)

    raise last_err


def process_one(row):
    """row = (claim_id, text, label, event). Label is metadata only — never sent to the LLM."""
    cid, text, label, event = row
    r_arg = generate_argument(text, RUMOUR)
    n_arg = generate_argument(text, NOT_RUMOUR)
    return {
        "claim_id": cid,
        "text": text,
        "label": int(label),
        "event": event,
        "rumour_argument": r_arg,
        "nonrumour_argument": n_arg,
        "qc_rumour": qc_check(r_arg),
        "qc_nonrumour": qc_check(n_arg),
    }


def load_done(path):
    """Return the set of claim_ids already generated, so a restart skips them."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["claim_id"])
            except (json.JSONDecodeError, KeyError):
                pass          # tolerate a half-written last line from a hard kill
    return done


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df = df.reset_index().rename(columns={"index": "claim_id"})
    print(f"loaded {len(df)} claims from {IN_CSV}")

    done = load_done(OUT_JSONL)
    todo = df[~df.claim_id.isin(done)]
    print(f"already done: {len(done)} | to generate: {len(todo)}")
    if len(todo) == 0:
        print("nothing to do.")
        return

    rows = list(todo[["claim_id", "text", "label", "event"]].itertuples(index=False, name=None))

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    qc_flagged = 0

    # newline="" + append: each result is durable the moment it lands
    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_one, r): r for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                except Exception as e:
                    n_fail += 1
                    print(f"  x claim {futures[fut][0]} failed: {type(e).__name__}: {e}")
                    continue

                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                n_ok += 1
                if rec["qc_rumour"] or rec["qc_nonrumour"]:
                    qc_flagged += 1

                if i % 50 == 0:
                    el = time.perf_counter() - t0
                    rate = i / el
                    eta = (len(rows) - i) / rate / 60
                    cost = _usage["in"] / 1e6 * PRICE_IN + _usage["out"] / 1e6 * PRICE_OUT
                    print(f"  {i}/{len(rows)} | {rate*60:.0f}/min | ETA {eta:.0f} min "
                          f"| ${cost:.2f} | QC flagged {qc_flagged}")

    el = time.perf_counter() - t0
    cost = _usage["in"] / 1e6 * PRICE_IN + _usage["out"] / 1e6 * PRICE_OUT
    avg_out = _usage["out"] / max(_usage["calls"], 1)

    print("\n" + "=" * 60)
    print(f"done: {n_ok} ok, {n_fail} failed, in {el/60:.1f} min")
    print(f"tokens: {_usage['in']:,} in / {_usage['out']:,} out over {_usage['calls']:,} calls")
    print(f"avg output tokens per call: {avg_out:.0f}")
    if avg_out > 600:
        print("  !! WARNING: output is far larger than 3-4 sentences should need.")
        print("     thinking is probably still ON (are you on an M2.x model?). Check before paying for the rest.")
    print(f"cost: ${cost:.2f}")
    print(f"QC flagged: {qc_flagged}/{n_ok} claims had at least one side flagged")
    print(f"-> {OUT_JSONL}")
    if n_fail:
        print(f"\n{n_fail} failed. Just re-run this script — it will retry only the missing ones.")


if __name__ == "__main__":
    main()