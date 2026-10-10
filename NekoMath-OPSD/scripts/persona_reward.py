"""Strict answer verification, async persona judging, and crash-safe rollout logs."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import aiohttp

ANSWER_RE = re.compile(r'<answer\s*>(.*?)</answer\s*>', re.IGNORECASE | re.DOTALL)
OPEN_RE = re.compile(r'<answer\b[^>]*>', re.IGNORECASE)
CLOSE_RE = re.compile(r'</answer\s*>', re.IGNORECASE)
THINK_RE = re.compile(r'<think\s*>(.*?)</think\s*>', re.IGNORECASE | re.DOTALL)
SCORE_KEYS = ('persona_consistency', 'naturalness', 'technical_clarity', 'overacting', 'repetition')


class JudgeParseError(ValueError):
    """The Judge returned a response but violated the required JSON contract."""


def canonical_text(value: Any) -> str:
    """Safe whitespace/unicode canonicalization; never changes mathematical meaning."""
    return ' '.join(unicodedata.normalize('NFKC', str(value)).strip().split())


def parse_strict_answer(completion: str) -> dict[str, Any]:
    """Accept exactly one nonempty, well-formed answer tag and nothing ambiguous."""
    opens = OPEN_RE.findall(completion)
    closes = CLOSE_RE.findall(completion)
    matches = ANSWER_RE.findall(completion)
    if not opens and not closes:
        return {'parse_success': False, 'format_valid': False, 'reason': 'missing_tag', 'answer': None}
    if len(opens) > 1 or len(closes) > 1 or len(matches) > 1:
        return {'parse_success': False, 'format_valid': False, 'reason': 'multiple_tag', 'answer': None}
    if len(opens) != 1 or len(closes) != 1 or len(matches) != 1:
        return {'parse_success': False, 'format_valid': False, 'reason': 'malformed_tag', 'answer': None}
    answer = canonical_text(matches[0])
    if not answer or '<' in answer or '>' in answer:
        return {'parse_success': False, 'format_valid': False, 'reason': 'malformed_tag', 'answer': None}
    return {'parse_success': True, 'format_valid': True, 'reason': 'ok', 'answer': answer}


def _safe_expr_equivalent(left: str, right: str) -> bool:
    """Optional numeric equivalence on a deliberately tiny arithmetic grammar."""
    allowed = re.compile(r'^[0-9+\-*/().\s^]+$')
    left, right = canonical_text(left), canonical_text(right)
    if not allowed.fullmatch(left) or not allowed.fullmatch(right):
        return False
    try:
        import sympy
        a = sympy.sympify(left.replace('^', '**'), locals={}, evaluate=True)
        b = sympy.sympify(right.replace('^', '**'), locals={}, evaluate=True)
        return bool(sympy.simplify(a - b) == 0)
    except Exception:
        return False


def answers_equivalent(answer: str, ground_truth: str) -> bool:
    """Strict-but-useful math equivalence with no execution of model text."""
    a, b = canonical_text(answer), canonical_text(ground_truth)
    if not a or not b:
        return False
    if a.casefold() == b.casefold():
        return True
    # Dataset-compatible LaTex canonicalizer (pure string operations).
    try:
        from verl.utils.reward_score.math_reward import is_equiv
        if is_equiv(a, b):
            return True
    except Exception:
        pass
    try:
        if Fraction(a) == Fraction(b):
            return True
    except Exception:
        pass
    return _safe_expr_equivalent(a, b)


def split_completion(completion: str) -> tuple[str, str]:
    match = THINK_RE.search(completion)
    if not match:
        return '', completion
    thinking = match.group(1)
    response = completion[:match.start()] + completion[match.end():]
    return thinking, response


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def persona_score(scores: dict[str, int]) -> float:
    p, n, c = scores['persona_consistency'] / 4, scores['naturalness'] / 4, scores['technical_clarity'] / 4
    o, r = scores['overacting'] / 4, scores['repetition'] / 4
    return max(0.0, min(1.0, 0.4 * p + 0.3 * n + 0.3 * c - 0.2 * o - 0.2 * r))


@dataclass
class JudgeStats:
    call_count: int = 0
    judged_completions: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    latency_seconds: float = 0.0
    parse_errors: int = 0
    request_errors: int = 0
    retries: int = 0

    def merge(self, other: 'JudgeStats') -> None:
        self.call_count += other.call_count
        self.judged_completions += other.judged_completions
        self.latency_seconds += other.latency_seconds
        self.parse_errors += other.parse_errors
        self.request_errors += other.request_errors
        self.retries += other.retries
        self.input_tokens = None if self.input_tokens is None or other.input_tokens is None else self.input_tokens + other.input_tokens
        self.output_tokens = None if self.output_tokens is None or other.output_tokens is None else self.output_tokens + other.output_tokens


class PersonaJudge:
    def __init__(self, prompt_path: str, base_url: str, model: str, timeout_seconds: float, retries: int, concurrency: int):
        self.template = Path(prompt_path).read_text(encoding='utf-8')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.sem = asyncio.Semaphore(concurrency)
        self.api_key = os.environ.get('STEPFUN_API_KEY')
        if not self.api_key:
            raise RuntimeError('STEPFUN_API_KEY is required for Persona Judge; it is never stored in artifacts.')

    @staticmethod
    def _strict_json(content: str) -> dict[str, int]:
        try:
            obj = json.loads(content.strip())
        except (TypeError, json.JSONDecodeError) as exc:
            raise JudgeParseError('response is not valid JSON') from exc
        if not isinstance(obj, dict) or set(obj) != set(SCORE_KEYS):
            raise JudgeParseError('judge JSON fields must be exactly the required five')
        if any(type(obj[k]) is not int or not 0 <= obj[k] <= 4 for k in SCORE_KEYS):
            raise JudgeParseError('judge values must be integer 0..4')
        return obj

    async def judge_one(self, session: aiohttp.ClientSession, problem: str, completion: str) -> tuple[dict[str, int] | None, JudgeStats, str | None]:
        prompt = self.template.replace('{{problem}}', problem).replace('{{completion}}', completion)
        stats = JudgeStats(judged_completions=1)
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'top_p': 1}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        async with self.sem:
            for attempt in range(self.retries + 1):
                started = time.monotonic()
                stats.call_count += 1
                try:
                    async with session.post(f'{self.base_url}/chat/completions', headers=headers, json=payload) as response:
                        body = await response.json(content_type=None)
                        response.raise_for_status()
                    stats.latency_seconds += time.monotonic() - started
                    usage = body.get('usage') or {}
                    if 'prompt_tokens' in usage and 'completion_tokens' in usage:
                        stats.input_tokens = (stats.input_tokens or 0) + int(usage['prompt_tokens'])
                        stats.output_tokens = (stats.output_tokens or 0) + int(usage['completion_tokens'])
                    else:
                        stats.input_tokens = stats.output_tokens = None
                    content = body['choices'][0]['message']['content']
                    return self._strict_json(content), stats, None
                except Exception as exc:
                    stats.latency_seconds += time.monotonic() - started
                    # ``judge_parse_error`` has a narrow, auditable meaning:
                    # a successful API response whose content violated the JSON
                    # contract.  Timeouts/HTTP/network issues are tracked apart.
                    if isinstance(exc, JudgeParseError):
                        stats.parse_errors += 1
                    else:
                        stats.request_errors += 1
                    if attempt >= self.retries:
                        return None, stats, f'{type(exc).__name__}: {exc}'
                    stats.retries += 1
                    await asyncio.sleep(min(2 ** attempt, 4))
        return None, stats, 'unexpected_judge_exit'

    async def judge_many(self, candidates: list[tuple[str, str]]) -> tuple[list[dict[str, int] | None], JudgeStats, list[str | None]]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(*(self.judge_one(session, p, c) for p, c in candidates))
        scores, errors = [], []
        total = JudgeStats()
        for score, stats, error in results:
            scores.append(score); errors.append(error); total.merge(stats)
        return scores, total, errors


class JsonlWriter:
    """Append one JSON object at a time and fsync so interrupted RL runs remain auditable."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())


def category_counts(records: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(records), 1)
    categories = Counter(record['parse_reason'] for record in records)
    return {
        'strict_format_valid_rate': sum(record['format_valid'] for record in records) / n,
        'parse_success_rate': sum(record['parse_success'] for record in records) / n,
        'missing_tag_rate': categories['missing_tag'] / n,
        'multiple_tag_rate': categories['multiple_tag'] / n,
        'malformed_tag_rate': categories['malformed_tag'] / n,
    }
