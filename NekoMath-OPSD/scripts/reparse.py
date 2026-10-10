#!/usr/bin/env python3
"""Re-parse already-saved judge replies that the old parser could not read.

Some judges emit valid rubric JSON but wrap it in prose / `</think>` / code
fences, which made an earlier `extract_json_object` give up. This walks the
output dirs, re-runs the (now hardened) parser on the stored `judge_raw`, and
appends a corrected row for every record that becomes usable. Because
`summarize.py` keeps the first usable row per key, the repaired row wins.

Nothing is deleted: the original rows stay in the file for audit.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import eval_persona as ep


def kind_of(row):
    kind = row.get("prompt_kind")
    if kind in ("correct", "failed"):
        return kind
    return "correct" if row.get("source_correct") else "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="judge output dirs (default: every judge_out* dir here)")
    ap.add_argument("--apply", action="store_true",
                    help="append repaired rows (default is a dry run)")
    ap.add_argument("--backup", action="store_true",
                    help="also copy each touched file to <name>.jsonl.bak first")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dirs = args.dirs or sorted(
        d for d in glob.glob(os.path.join(here, "judge_out*")) if os.path.isdir(d))

    grand_fixed = grand_bad = grand_missing = 0
    for d in dirs:
        paths = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        per_file = {}
        usable = set()
        for path in paths:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
            per_file[path] = rows
            for row in rows:
                if row.get("schema_ok"):
                    usable.add((row.get("arm"), row.get("index"), row.get("sample_id")))
        rows_total = sum(len(r) for r in per_file.values())
        already_ok = len(usable)

        repaired = {}          # key -> (path, row)  needed to close a gap
        repairable = 0         # any row the new parser can now read
        still_bad = raw_missing = 0
        for path, rows in per_file.items():
            for row in rows:
                if row.get("schema_ok"):
                    continue
                raw = row.get("judge_raw")
                if not raw:
                    raw_missing += 1
                    still_bad += 1
                    continue
                kind = kind_of(row)
                obj = ep.extract_json_object(raw, kind)
                parsed, ok = ep.validate(obj, kind)
                if not ok:
                    still_bad += 1
                    continue
                repairable += 1
                key = (row.get("arm"), row.get("index"), row.get("sample_id"))
                if key in usable:
                    continue  # a later retry already produced a usable row
                new = dict(row)
                new["judge_parsed"] = parsed
                new["schema_ok"] = True
                new["judge_error"] = None
                new["judge_error_before_reparse"] = row.get("judge_error")
                new["judge_reparsed"] = True
                new["judge_reparsed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                repaired[key] = (path, new)   # last repairable row wins

        print("=" * 68)
        print("%s" % d)
        print("  rows=%d usable_keys=%d  repairable_rows=%d  need_reparse=%d  "
              "still_bad=%d (raw missing=%d)"
              % (rows_total, already_ok, repairable, len(repaired), still_bad, raw_missing))

        if args.apply and repaired:
            by_path = {}
            for path, row in repaired.values():
                by_path.setdefault(path, []).append(row)
            for path, new_rows in by_path.items():
                if args.backup:
                    shutil.copyfile(path, path + ".bak")
                with open(path, "a", encoding="utf-8") as f:
                    for row in new_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("  appended %d repaired row(s) -> %s" % (len(new_rows), path))

        grand_fixed += len(repaired)
        grand_bad += still_bad
        grand_missing += len(repaired)

    print("=" * 68)
    print("TOTAL: %d rows repaired, %d still unusable%s"
          % (grand_fixed, grand_bad, "" if args.apply else " (dry run; pass --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
