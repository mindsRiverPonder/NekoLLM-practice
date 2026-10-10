#!/usr/bin/env python3
"""Cross-judge agreement report.

Compares two or more judge output directories record-by-record on the persona
rubric. Reports per-dimension correlation, agreement rates, and whether the
five arms are ranked the same way by each judge.
"""
import argparse, glob, gzip, json, os, statistics as st
from collections import defaultdict

DIMS = ["persona_consistency", "character_specificity", "naturalness",
        "technical_clarity", "overacting", "repetition"]
REASON = ["reasoning_correctness", "solution_progress", "answer_proximity",
          "error_recoverability"]


def load(d):
    best = {}
    for fn in glob.glob(os.path.join(d, "*.jsonl")) + glob.glob(os.path.join(d, "*.jsonl.gz")):
        arm = os.path.basename(fn).removesuffix(".gz").removesuffix(".jsonl")
        for line in (gzip.open if fn.endswith(".gz") else open)(fn, "rt", encoding="utf-8"):
            r = json.loads(line)
            k = (arm, r["index"], r["sample_id"])
            cur = best.get(k)
            if cur is None or (r.get("schema_ok") and not cur.get("schema_ok")):
                best[k] = r
    return best


def ranks(vals_by_arm):
    order = sorted(vals_by_arm, key=lambda a: -vals_by_arm[a])
    return {a: i + 1 for i, a in enumerate(order)}, order


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="judge output dirs")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--baseline", type=int, default=0, help="index of reference judge")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.dirs]
    data = [load(d) for d in args.dirs]
    base = data[args.baseline]

    for j, (lab, other) in enumerate(zip(labels, data)):
        if j == args.baseline:
            continue
        keys = [k for k in base
                if k in other and base[k]["schema_ok"] and other[k]["schema_ok"]]
        print("=" * 78)
        print("judge A = %s   vs   judge B = %s" % (labels[args.baseline], lab))
        print("paired records: %d" % len(keys))
        print("=" * 78)
        print("%-22s %7s %7s %8s %8s %8s" %
              ("dimension", "pearson", "spearman", "exact%", "<=1%", "meanB-A"))
        for dim in DIMS + REASON:
            pairs = [(base[k]["judge_parsed"].get(dim), other[k]["judge_parsed"].get(dim))
                     for k in keys]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) < 30:
                continue
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            d = [b - a for a, b in pairs]
            print("%-22s %7.3f %8.3f %7.1f%% %7.1f%% %+8.2f" %
                  (dim, pearson(xs, ys), spearman(xs, ys),
                   100 * sum(1 for x, y in pairs if x == y) / len(pairs),
                   100 * sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs),
                   st.mean(d)))
        print()
        print("arm ranking per dimension (best -> worst):")
        for dim in DIMS:
            cols = []
            for name, d in zip(labels, data):
                means = defaultdict(list)
                for k, r in d.items():
                    if r["schema_ok"] and dim in r["judge_parsed"]:
                        means[k[0]].append(r["judge_parsed"][dim])
                if not means:
                    continue
                mv = {a: st.mean(v) for a, v in means.items()}
                _, order = ranks(mv)
                cols.append((name, order))
            print("  %-22s %s" % (dim, "   |   ".join(
                "[%s] %s" % (n, " > ".join(o)) for n, o in cols)))
        print()


if __name__ == "__main__":
    main()
