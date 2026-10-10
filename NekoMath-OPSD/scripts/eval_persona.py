#!/usr/bin/env python3
"""External persona-judge runner for the persona_eval_bundle.

Routes each rollout to the correct/failed prompt template, calls an
OpenAI-compatible chat endpoint as the judge, validates the returned JSON
rubric, and appends one result record per rollout to <out-dir>/<arm>.jsonl.

Resume-safe: records whose (arm, index, sample_id) already exist in the
output file are skipped, so the script can be restarted at any time.
"""
import argparse
import gzip
import collections
import concurrent.futures as cf
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

CORRECT_FIELDS = [
    "persona_consistency", "character_specificity", "naturalness",
    "technical_clarity", "overacting", "repetition",
]
FAILED_EXTRA_FIELDS = [
    "reasoning_correctness", "solution_progress", "answer_proximity",
    "error_recoverability",
]
SCORE_FIELDS = CORRECT_FIELDS + FAILED_EXTRA_FIELDS
ERROR_TYPES = {
    "arithmetic", "sign_or_transcription", "algebraic", "logical", "condition",
    "wrong_method", "incomplete", "truncated", "unsupported_jump",
    "multiple_errors", "other",
}
ERROR_STAGES = {"early", "middle", "late", "final_only", "unclear"}


class FatalJudgeError(RuntimeError):
    """Raised when the endpoint looks permanently broken (e.g. revoked key)."""


# ---------------------------------------------------------------- throttling

class RateLimiter:
    """Limits call starts to `rpm` per rolling minute.

    With `even=True` the window is *paced* instead of bursty: starts are spread
    at least 60/rpm seconds apart, so a strict per-minute cap on the server side
    can never see a spike (safer against abuse detection).
    """

    def __init__(self, rpm, even=False):
        self.rpm = rpm
        self.even = even
        self.lock = threading.Lock()
        self.stamps = collections.deque()
        self.blocked_until = 0.0
        self.next_slot = 0.0

    def acquire(self):
        if self.rpm <= 0:
            return
        while True:
            with self.lock:
                now = time.time()
                while self.stamps and now - self.stamps[0] > 60.0:
                    self.stamps.popleft()
                wait = 0.0
                if now < self.blocked_until:
                    wait = self.blocked_until - now
                elif len(self.stamps) >= self.rpm:
                    wait = 60.0 - (now - self.stamps[0]) + 0.05
                elif self.even and now < self.next_slot:
                    wait = self.next_slot - now
                if wait <= 0:
                    self.stamps.append(now)
                    if self.even:
                        self.next_slot = now + 60.0 / self.rpm
                    return
            time.sleep(min(max(wait, 0.05), 30.0))

    def block_for(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.time() + seconds)

    def note_reject(self):
        """A 429 was observed: shrink the local window so we pace down."""
        with self.lock:
            now = time.time()
            while self.stamps and now - self.stamps[0] > 60.0:
                self.stamps.popleft()
            while len(self.stamps) < self.rpm:
                self.stamps.appendleft(now)


def parse_429_reset(error_text):
    """Extract the reset timestamp from a LiteLLM throttling error."""
    m = re.search(r"Limit resets at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})",
                  error_text or "")
    if not m:
        return None
    try:
        when = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        return None
    delta = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    return max(delta, 1.0)


# ---------------------------------------------------------------- utilities

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(template, mapping):
    out = template
    for key, value in mapping.items():
        out = out.replace("{{%s}}" % key, value)
    return out


def _expected_fields(kind):
    if kind == "failed":
        return SCORE_FIELDS + ["primary_error_type", "error_stage"]
    if kind == "correct":
        return CORRECT_FIELDS
    return SCORE_FIELDS


def _json_objects(text):
    """Yield (offset, obj) for every strict JSON object embedded in `text`."""
    if not text:
        return
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        pos = text.find("{", idx)
        if pos == -1:
            return
        try:
            obj, _end = decoder.raw_decode(text, pos)
        except Exception:
            idx = pos + 1
            continue
        if isinstance(obj, dict):
            yield pos, obj
        idx = pos + 1


def extract_json_object(text, kind=None):
    """Pull the best judge JSON object out of a (possibly chatty) reply.

    Some judges reason in plain text and only emit the rubric JSON at the very
    end, sometimes after a `</think>` tag or a code fence. Strategy:

      1. strict parse of the whole (fence-stripped) reply -- the happy path;
      2. the legacy first-`{`-to-last-`}` slice with light repairs;
      3. scan every `{` with `raw_decode` and keep the object that best matches
         the rubric, preferring the last one on ties (the answer comes last).

    `kind` ('correct'/'failed') lets the scorer know which fields to look for.
    """
    if not text:
        return None
    expected = _expected_fields(kind)
    full = len(expected)

    best = [None, (-1, -1)]

    def consider(obj, pos):
        if not isinstance(obj, dict):
            return
        key = (sum(1 for f in expected if f in obj), pos)
        if key > best[1]:
            best[0], best[1] = obj, key

    s = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", s)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    for blob in (stripped, s):
        try:
            consider(json.loads(blob), 0)
        except Exception:
            pass
    if best[1][0] == full:
        return best[0]

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        candidate = s[start:end + 1]
        for repaired in (candidate,
                         re.sub(r",\s*([}\]])", r"\1", candidate.replace("'", '"'))):
            try:
                consider(json.loads(repaired), start)
            except Exception:
                pass
    if best[1][0] == full:
        return best[0]

    for pos, obj in _json_objects(s):
        consider(obj, pos)
    return best[0]


def validate(obj, kind):
    """Normalise and validate a judge JSON object. Returns (parsed, schema_ok)."""
    if not isinstance(obj, dict):
        return None, False
    expected = list(CORRECT_FIELDS)
    if kind == "failed":
        expected = list(SCORE_FIELDS)
    parsed = {}
    ok = True
    for field in expected:
        if field not in obj:
            ok = False
            continue
        value = obj[field]
        if isinstance(value, bool):
            ok = False
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, str) and re.fullmatch(r"\d", value.strip()):
            value = int(value.strip())
        if not isinstance(value, int) or not 0 <= value <= 4:
            ok = False
            continue
        parsed[field] = value
    expected_keys = list(expected)
    if kind == "failed":
        expected_keys += ["primary_error_type", "error_stage"]
    extra = sorted(set(obj) - set(expected_keys))
    if extra:
        ok = False
        parsed["_extra_fields"] = extra
    if kind == "failed":
        for field, allowed in (("primary_error_type", ERROR_TYPES),
                               ("error_stage", ERROR_STAGES)):
            value = obj.get(field)
            if isinstance(value, str) and value.strip() in allowed:
                parsed[field] = value.strip()
            else:
                ok = False
    if len(parsed) < len(expected):
        ok = False
    return parsed, ok


def post_json(url, payload, api_key, timeout, user_agent=None, extra_headers=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + api_key}
    if user_agent:
        headers["User-Agent"] = user_agent
    for name, value in (extra_headers or {}).items():
        headers[name] = value
    req = urllib.request.Request(
        url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        return None, "HTTP %s: %s" % (e.code, detail)
    except Exception as e:  # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, e)


# ---------------------------------------------------------------- judge call

class Judge:
    def __init__(self, args):
        self.args = args
        self.limiter = args.limiter
        self.no_thinking = args.no_thinking
        self.stop_event = threading.Event()
        self.auth_lock = threading.Lock()
        self.consecutive_auth_failures = 0
        base = args.base_url.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.templates = {}
        for kind, name in (("correct", "persona_correct.txt"),
                           ("failed", "persona_failed.txt")):
            path = os.path.join(args.prompt_dir, name)
            with open(path, encoding="utf-8") as f:
                self.templates[kind] = f.read()
        self.model_tag = args.model

    def build_messages(self, record):
        kind = "correct" if record.get("correct") else "failed"
        mapping = {
            "problem": record["problem"],
            "completion": record["completion"],
        }
        if kind == "failed":
            mapping["reference_answer"] = record.get("ground_truth") or ""
        prompt = render(self.templates[kind], mapping)
        return kind, [{"role": "user", "content": prompt}]

    def judge_one(self, record):
        args = self.args
        if self.stop_event.is_set():
            raise FatalJudgeError("aborted")
        kind, messages = self.build_messages(record)
        max_tokens = args.max_tokens
        last_error = None
        retry_log = []
        started = time.time()
        for attempt in range(1, args.attempts + 1):
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "stream": False,
            }
            if args.top_p is not None:
                payload["top_p"] = args.top_p
            if args.seed is not None:
                payload["seed"] = args.seed
            if self.no_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            self.limiter.acquire()
            data, error = post_json(self.url, payload, args.api_key, args.timeout,
                                    args.user_agent, args.extra_headers)
            if data is None:
                last_error = error
                if re.match(r"HTTP 401\b", error) or "Invalid token" in error:
                    self._note_auth_failure(args, error, retry_log)
                    continue
                if "HTTP 429" in error:
                    self.limiter.note_reject()
                    reset = parse_429_reset(error)
                    self.limiter.block_for(min(reset or 20.0, args.throttle_block))
                    retry_log.append({"attempt": attempt, "reason": "429",
                                      "reset_s": round(reset or 0.0, 1)})
                    if attempt == args.attempts:
                        break
                    time.sleep(args.throttle_sleep + random.random() * args.throttle_jitter)
                    continue
                # 403 (Cloudflare bot check), 408, 5xx and transport errors are worth retrying
                retryable = not re.match(r"HTTP (400|401|404|422)\b", error)
                if not retryable or attempt == args.attempts:
                    break
                retry_log.append({"attempt": attempt, "reason": error[:120]})
                time.sleep(min(args.backoff * 2 ** (attempt - 1), 30) + random.random())
                continue
            if "choices" not in data or not data["choices"]:
                detail = json.dumps(data)[:300]
                if "429" in detail or "throttling" in detail:
                    self.limiter.note_reject()
                    self.limiter.block_for(args.throttle_block)
                retry_log.append({"attempt": attempt, "reason": "no-choices: %s" % detail[:120]})
                last_error = "no choices: %s" % detail
                time.sleep(min(args.backoff * 2 ** (attempt - 1), 30) + random.random())
                continue
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            finish = choice.get("finish_reason")
            obj = extract_json_object(content, kind)
            parsed, schema_ok = validate(obj, kind)
            truncated = finish == "length"
            if (parsed is None or not schema_ok) and truncated and max_tokens < args.max_tokens_ceiling:
                # reasoning ate the budget: retry with a larger cap
                max_tokens = min(max_tokens * 2, args.max_tokens_ceiling)
                last_error = "truncated, retry with max_tokens=%d" % max_tokens
                retry_log.append({"attempt": attempt, "reason": "truncated->%d" % max_tokens})
                continue
            if parsed is None and attempt < args.attempts:
                last_error = "unparseable judge output"
                retry_log.append({"attempt": attempt, "reason": "unparseable"})
                time.sleep(min(args.backoff * 2 ** (attempt - 1), 30) + random.random())
                continue
            usage = data.get("usage") or {}
            self.note_success()
            return {
                "prompt_kind": kind,
                "judge_raw": content,
                "judge_parsed": parsed,
                "schema_ok": schema_ok,
                "judge_finish_reason": finish,
                "judge_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "judge_usage": usage,
                "judge_error": None if schema_ok else (last_error or "schema mismatch"),
                "judge_attempts": attempt,
                "judge_retry_log": retry_log,
                "judge_max_tokens_used": max_tokens,
                "judge_latency_s": round(time.time() - started, 3),
            }
        return {
            "prompt_kind": kind,
            "judge_raw": None,
            "judge_parsed": None,
            "schema_ok": False,
            "judge_finish_reason": None,
            "judge_reasoning_tokens": None,
            "judge_usage": None,
            "judge_error": last_error or "failed",
            "judge_attempts": args.attempts,
            "judge_retry_log": retry_log,
            "judge_max_tokens_used": max_tokens,
            "judge_latency_s": round(time.time() - started, 3),
        }

    def _note_auth_failure(self, args, error, retry_log):
        """A revoked/expired key fails every request: stop instead of burning the queue."""
        with self.auth_lock:
            self.consecutive_auth_failures += 1
            count = self.consecutive_auth_failures
        retry_log.append({"attempt": 1, "reason": "auth: %s" % error[:100]})
        if count >= args.abort_after_auth_failures:
            self.stop_event.set()
            raise FatalJudgeError(
                "endpoint rejected the API key %d times in a row (%s); aborting run"
                % (count, error[:120]))
        time.sleep(2.0)

    def note_success(self):
        if self.consecutive_auth_failures:
            with self.auth_lock:
                self.consecutive_auth_failures = 0


# ---------------------------------------------------------------- driver

def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge_error") is None:
                done.add((r["index"], r["sample_id"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=os.path.dirname(HERE))
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(HERE), "outputs", "judge_out"))
    p.add_argument("--prompt-dir", default=os.path.join(os.path.dirname(HERE), "prompts", "evaluation"))
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="max records per arm (0 = all)")
    p.add_argument("--per-arm", type=int, default=0, help="alias of --limit")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-tokens-ceiling", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--attempts-on-throttle", type=int, default=0,
                   help="extra attempts reserved for rate-limit stalls (0 = auto)")
    p.add_argument("--rpm", type=float, default=19.0,
                   help="client-side request cap per minute (server limit is 20)")
    p.add_argument("--throttle-block", type=float, default=8.0,
                   help="max seconds a single 429 stalls all workers")
    p.add_argument("--throttle-sleep", type=float, default=3.0)
    p.add_argument("--throttle-jitter", type=float, default=5.0)
    p.add_argument("--even-pacing", action="store_true",
                   help="spread requests at 60/rpm intervals instead of bursting")
    p.add_argument("--no-thinking", action="store_true",
                   help="disable judge reasoning (much faster, lower quality risk)")
    p.add_argument("--user-agent", default=None,
                   help="User-Agent header (some gateways block python's default)")
    p.add_argument("--extra-header", action="append", default=[], metavar="NAME:VALUE",
                   help="extra request header, repeatable (e.g. ngrok-skip-browser-warning:any)")
    p.add_argument("--abort-after-auth-failures", type=int, default=25,
                   help="abort the whole run after this many consecutive 401s")
    p.add_argument("--backoff", type=float, default=2.0)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--verify-hashes", action="store_true")
    args = p.parse_args()
    if args.per_arm:
        args.limit = args.per_arm
    if not args.api_key:
        sys.exit("missing --api-key (or JUDGE_API_KEY env var)")
    if args.rpm and args.rpm > 0:
        # throttling stalls eat attempts; give the judge enough rope
        args.attempts = max(args.attempts, 6)
    args.limiter = RateLimiter(args.rpm, even=args.even_pacing)
    args.extra_headers = {}
    for raw in args.extra_header:
        if ":" not in raw:
            sys.exit("--extra-header expects NAME:VALUE, got %r" % raw)
        name, value = raw.split(":", 1)
        args.extra_headers[name.strip()] = value.strip()

    manifest_path = os.path.join(args.bundle, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    arms = args.arms or sorted(manifest["arms"])

    if args.verify_hashes:
        for arm in arms:
            info = manifest["arms"][arm]
            got = sha256_file(os.path.join(args.bundle, info["file"]))
            status = "OK" if got == info["sha256"] else "MISMATCH"
            print("[hash] %-16s %s" % (arm, status))
            if status != "OK": sys.exit("Rollout hash mismatch: " + arm)

    os.makedirs(args.out_dir, exist_ok=True)
    judge = Judge(args)
    lock = threading.Lock()

    handles = {}
    stats = {}
    pending_by_arm = {}
    for arm in arms:
        src_path = os.path.join(args.bundle, manifest["arms"][arm]["file"])
        out_path = os.path.join(args.out_dir, arm + ".jsonl")
        done = load_done(out_path)
        pending = []
        with (gzip.open if src_path.endswith(".gz") else open)(src_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i < args.offset:
                    continue
                if args.limit and len(pending) >= args.limit:
                    break
                record = json.loads(line)
                if (record["index"], record["sample_id"]) in done:
                    continue
                pending.append(record)
        pending_by_arm[arm] = pending
        stats[arm] = Counter()
        handles[arm] = open(out_path, "a", encoding="utf-8")
        if done:
            print("[%s] resuming, %d already judged" % (arm, len(done)))

    # Interleave arms so partial progress stays balanced across the whole set.
    tasks = []
    i = 0
    while True:
        added = False
        for arm in arms:
            pending = pending_by_arm[arm]
            if i < len(pending):
                tasks.append((arm, pending[i]))
                added = True
        if not added:
            break
        i += 1

    total_tasks = len(tasks)
    print("total pending: %d records across %d arms (concurrency=%d rpm=%g)"
          % (total_tasks, len(arms), args.concurrency, args.rpm))
    grand_total = grand_done = grand_fail = 0
    if total_tasks:
        t0 = time.time()
        done_n = 0
        aborted = False
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(judge.judge_one, rec): (arm, rec) for arm, rec in tasks}
            for fut in cf.as_completed(futures):
                arm, record = futures[fut]
                try:
                    judged = fut.result()
                except FatalJudgeError as e:
                    if not aborted:
                        aborted = True
                        judge.stop_event.set()
                        print("  !! %s" % e, flush=True)
                        print("  !! stopping: %d/%d records written, re-run to resume"
                              % (done_n, total_tasks), flush=True)
                    continue
                except Exception as e:  # noqa: BLE001
                    judged = {"prompt_kind": "?", "judge_raw": None, "judge_parsed": None,
                              "schema_ok": False,
                              "judge_error": "%s: %s" % (type(e).__name__, e),
                              "judge_attempts": 0, "judge_retry_log": []}
                row = {
                    "arm": arm,
                    "index": record["index"],
                    "sample_id": record["sample_id"],
                    "judge_model": args.model,
                    "judge_temperature": args.temperature,
                    "source_correct": record.get("correct"),
                    "source_format_valid": record.get("format_valid"),
                    "source_parse_success": record.get("parse_success"),
                    "source_extracted_answer": record.get("extracted_answer"),
                    "reference_answer": record.get("ground_truth"),
                    "source_completion_tokens": record.get("completion_tokens"),
                    "source_finish_reason": record.get("finish_reason"),
                }
                row.update(judged)
                with lock:
                    handles[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
                    done_n += 1
                    stats[arm]["ok" if judged.get("schema_ok") else "bad"] += 1
                    if done_n % args.checkpoint_every == 0:
                        for h in handles.values():
                            h.flush()
                        os.fsync(handles[arm].fileno())
                if done_n % 25 == 0 or done_n == total_tasks:
                    elapsed = time.time() - t0
                    rate = done_n / elapsed if elapsed else 0
                    eta = (total_tasks - done_n) / rate if rate else 0
                    ok = sum(s["ok"] for s in stats.values())
                    print("  %d/%d | ok=%d bad=%d | %.2f rec/min | eta %.1f h"
                          % (done_n, total_tasks, ok, done_n - ok, rate * 60, eta / 3600),
                          flush=True)
            if aborted:
                for f in futures:
                    f.cancel()

    summary = {}
    for arm in arms:
        written = stats[arm]["ok"] + stats[arm]["bad"]
        summary[arm] = {"judged": written,
                        "pending": len(pending_by_arm[arm]),
                        "schema_ok": stats[arm]["ok"],
                        "needs_review": stats[arm]["bad"]}
        grand_total += written
        grand_done += stats[arm]["ok"]
        grand_fail += stats[arm]["bad"]
        handles[arm].flush()
        handles[arm].close()

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("total judged=%d schema_ok=%d needs_review=%d" % (grand_total, grand_done, grand_fail))


if __name__ == "__main__":
    main()
