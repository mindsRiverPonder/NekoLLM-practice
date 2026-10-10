#!/usr/bin/env python3
"""Aggregate judge outputs into per-arm score tables."""
import argparse, glob, gzip, json, os, statistics as st
from collections import defaultdict

PERSONA = ["persona_consistency", "character_specificity", "naturalness",
           "technical_clarity", "overacting", "repetition"]
REASON = ["reasoning_correctness", "solution_progress", "answer_proximity",
          "error_recoverability"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_out"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--raw", action="store_true",
                    help="count every line instead of deduplicating reruns by key")
    args = ap.parse_args()

    arms = {}
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*.jsonl")) + glob.glob(os.path.join(args.out_dir, "*.jsonl.gz"))):
        arm = os.path.basename(path).removesuffix(".gz").removesuffix(".jsonl")
        with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(l) for l in handle if l.strip()]
        if not args.raw:
            # a record may appear several times when a failed run is resumed;
            # keep the first usable judgement per (index, sample_id)
            best = {}
            for r in rows:
                key = (r["index"], r["sample_id"])
                cur = best.get(key)
                if cur is None or (r.get("schema_ok") and not cur.get("schema_ok")):
                    best[key] = r
            rows = list(best.values())
        arms[arm] = rows

    print("%-16s %7s %6s %6s | %s" % ("arm", "n", "ok%", "corr%", "persona means (0-4)"))
    csv_rows = []
    for arm, rows in arms.items():
        ok = [r for r in rows if r.get("schema_ok")]
        corr = [r for r in ok if r["prompt_kind"] == "correct"]
        fail = [r for r in ok if r["prompt_kind"] == "failed"]
        cells = []
        for f in PERSONA:
            vals = [r["judge_parsed"][f] for r in ok if f in r["judge_parsed"]]
            cells.append("%s=%.2f" % (f[:4], st.mean(vals)) if vals else "%s=--" % f[:4])
        print("%-16s %7d %5.1f%% %5.1f%% | %s"
              % (arm, len(rows), 100 * len(ok) / max(len(rows), 1),
                 100 * len(corr) / max(len(rows), 1), " ".join(cells)))
        for f in PERSONA + REASON:
            for kind, subset in (("all", ok), ("correct", corr), ("failed", fail)):
                vals = [r["judge_parsed"][f] for r in subset if f in r["judge_parsed"]]
                if vals:
                    csv_rows.append([arm, kind, f, len(vals), round(st.mean(vals), 4),
                                     round(st.pstdev(vals), 4)])
        for f, kind in (("primary_error_type", "label"), ("error_stage", "label")):
            counts = defaultdict(int)
            for r in fail:
                v = r["judge_parsed"].get(f)
                if v:
                    counts[v] += 1
            if counts:
                top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
                csv_rows.append([arm, "failed", f, sum(counts.values()),
                                 ";".join("%s:%d" % kv for kv in top), ""])

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("arm,subset,metric,n,mean,std\n")
            for row in csv_rows:
                fh.write(",".join(str(x) for x in row) + "\n")
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()
