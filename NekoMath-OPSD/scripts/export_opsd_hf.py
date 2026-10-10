#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get('COLDSTART_MODEL', '')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=BASE)
    p.add_argument('--state', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not a.base: p.error('Set --base or COLDSTART_MODEL')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    state = torch.load(a.state, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'load mismatch: missing={missing}, unexpected={unexpected}')
    model.save_pretrained(out, safe_serialization=True, max_shard_size='5GB')
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    tok.save_pretrained(out)
    print(f'exported {len(state)} tensors to {out}')

if __name__ == '__main__':
    main()
