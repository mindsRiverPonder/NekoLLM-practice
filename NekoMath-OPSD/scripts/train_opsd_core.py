#!/usr/bin/env python3
"""Neko-OPSD final trainer, used by seeds 42, 1234 and 5678.

Frozen coldstart teacher; reference in teacher system context; full-parameter
student; per-position 5*tanh(KL/5), then per-answer and batch averaging.
Release edits only parameterize input/model paths; optimization is unchanged.
"""
from __future__ import annotations
import argparse, datetime, json, os, random, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as ckpt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL = os.environ.get('COLDSTART_MODEL', '')
ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ''  # Initialized from --system-prompt after parsing arguments.


def templ(tok, messages):
    x = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    ids = x['input_ids'] if hasattr(x, '__getitem__') and not isinstance(x, (list, tuple)) else x
    if torch.is_tensor(ids):
        ids = ids.squeeze(0).tolist()
    elif hasattr(ids, 'ids'):
        ids = ids.ids
    elif isinstance(ids, (list, tuple)):
        if len(ids) == 1 and hasattr(ids[0], 'ids'):
            ids = ids[0].ids
        elif ids and hasattr(ids[0], 'ids'):
            ids = ids[0].ids
        elif ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
    return [int(v) for v in ids]


def pad_left(seqs, pad_id, device):
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, width - len(s):] = 1
    return ids, mask, width


def trim(ids, eos):
    ids = [int(v) for v in ids]
    return ids[:ids.index(eos) + 1] if eos is not None and eos in ids else ids


def privileged_system(completion):
    """System message of the privileged teacher: the only difference from the
    student is the reference solution.

    v2 put the reference in the *user* turn with `[REFERENCE_CORRECT_SOLUTION_*]`
    / `[STUDENT_CURRENT_PREFIX_START]` markers, so the teacher's next token was
    the continuation of that marker block and the student was trained to match
    it.  The finished run emitted 420 markers and 197 "参考解" mentions that
    appear nowhere in the data or in any other arm, while Pass@4 fell from 0.679
    (rft) to 0.511.  Keeping the reference in the system turn leaves the
    assistant-turn prefix byte-identical between the two prompts, which
    `row_prompts` now asserts.
    """
    return (SYSTEM
            + '\n\n下面是一份参考解答，供你核对思路；不要复述它，也不要提及它的存在。\n'
            + completion.strip())


def fit_tokens(ids, budget):
    """Keep head (system+question) and tail (end of reference+marker) of a prompt."""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head = budget // 2
    return ids[:head] + ids[-(budget - head):]


def gsum(x, dev):
    t = torch.tensor([float(x)], device=dev, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def pack_rows(lengths, budget):
    """Greedy packing: group rows so that padded_tokens = len(group) * max_len <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        cand = cur + [i]
        lmax = max(lengths[j] for j in cand)
        if cur and (budget <= 0 or lmax * len(cand) > budget):
            out.append(cur)
            cur = [i]
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def hidden_states(backbone, seqs, device, grad):
    """Left-padded batched backbone pass; returns one [L, H] view per sequence."""
    ids, mask, width = pad_left(seqs, 0, device)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    if grad:
        out = backbone(input_ids=ids, attention_mask=mask, position_ids=pos)
    else:
        with torch.no_grad():
            out = backbone(input_ids=ids, attention_mask=mask, position_ids=pos)
    h = out.last_hidden_state
    return [h[k, width - len(s):] for k, s in enumerate(seqs)]


def make_chunk_kl(s_head, t_head, kl_clip):
    """Forward KL(teacher || student) for one hidden-state chunk.

    Differentiable in the student hidden states only; the teacher side and the
    log-probabilities of the sampled tokens are computed under no_grad.  The
    per-token KL is passed through a smooth saturating surrogate so that large
    KL tokens keep a (shrunken) gradient instead of getting a hard zero.
    """
    t_dev = next(t_head.parameters()).device

    def chunk_kl(hs_chunk, ht_chunk):
        with torch.no_grad():
            ltp = torch.log_softmax(t_head(ht_chunk.to(t_dev)).float(), dim=-1)
            pt = ltp.exp()
        lsp = torch.log_softmax(s_head(hs_chunk).float(), dim=-1)
        kl = (pt * (ltp - lsp)).sum(-1)
        if kl_clip > 0:
            loss = kl_clip * torch.tanh(kl / kl_clip)
            nclip = (kl.detach() > kl_clip).sum()
        else:
            loss = kl
            nclip = torch.zeros((), device=kl.device, dtype=torch.long)
        return loss.sum(), kl.detach().sum(), nclip
    return chunk_kl


def sync_gradients(params, divisor, bucket_mb=128):
    """Average local gradients once per update and scale by the global sample count.

    Online rollouts have rank-dependent lengths, so DDP forward/backward inside
    the KL chunks is unsafe (ranks would execute different numbers of NCCL
    collectives).  Training therefore uses replicated parameters and this
    deterministic parameter-ordered reduction after all local work is done.
    `divisor` is the *global* number of samples that contributed a gradient, so
    an empty rollout slot no longer dilutes the update.
    """
    with torch.no_grad():
        bucket, bucket_bytes = [], 0
        limit = bucket_mb * 1024 * 1024

        def flush():
            nonlocal bucket, bucket_bytes
            if not bucket:
                return
            flat = torch.cat([p.grad.reshape(-1) for p in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(divisor)
            pos = 0
            for p in bucket:
                n = p.numel()
                p.grad.copy_(flat[pos:pos + n].view_as(p.grad))
                pos += n
            del flat
            bucket, bucket_bytes = [], 0

        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            nbytes = p.grad.numel() * p.grad.element_size()
            if bucket and bucket_bytes + nbytes > limit:
                flush()
            bucket.append(p)
            bucket_bytes += nbytes
        flush()


class ShardedMasterAdamW:
    """bf16 compute parameters + fp32 master shard + sharded Adam moments.

    A bf16 in-place update at lr<=1e-6 is smaller than the bf16 ULP for every
    weight with |w| >= ~3e-4, so the update is lost to rounding (measured above:
    0.00% of the large weights moved over 404 steps).  Accumulating in fp32 and
    rounding only once per step fixes that.  The master copy and the moments are
    split across ranks (ZeRO-1 style); after the local shard has been updated the
    owners broadcast their slices so every rank holds the full bf16 parameters.
    """

    tag = 'sharded-master-adamw-v1'
    CHUNK = 1 << 26  # elements per owned chunk (~256 MB in fp32)

    def __init__(self, params, world, rank, lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, master_dtype=torch.float32, moment_dtype=torch.bfloat16):
        self.params = list(params)
        self.world = world
        self.rank = rank
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.master_dtype = master_dtype
        self.moment_dtype = moment_dtype
        self.t = 0
        self.chunks = []  # (param_index, offset, length, owner_rank)
        for pi, p in enumerate(self.params):
            n = p.numel()
            off = 0
            while off < n:
                take = min(self.CHUNK, n - off)
                self.chunks.append((pi, off, take, len(self.chunks) % world))
                off += take
        self.owned = [i for i, c in enumerate(self.chunks) if c[3] == rank]
        # shards must live on the same device as the parameters they accumulate
        self.master = [torch.zeros(self.chunks[i][2], dtype=master_dtype,
                                   device=self.params[self.chunks[i][0]].device) for i in self.owned]
        self.m = [torch.zeros(self.chunks[i][2], dtype=moment_dtype,
                              device=self.params[self.chunks[i][0]].device) for i in self.owned]
        self.v = [torch.zeros(self.chunks[i][2], dtype=moment_dtype,
                              device=self.params[self.chunks[i][0]].device) for i in self.owned]
        self.sync_master_from_params()

    def _view(self, pi, off, length):
        return self.params[pi].data.view(-1)[off:off + length]

    @torch.no_grad()
    def sync_master_from_params(self):
        for i, ci in enumerate(self.owned):
            pi, off, n, _ = self.chunks[ci]
            self.master[i].copy_(self._view(pi, off, n).float())

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, ci in enumerate(self.owned):
            pi, off, n, _ = self.chunks[ci]
            p = self.params[pi]
            g = p.grad
            gv = g.view(-1)[off:off + n].float() if g is not None else torch.zeros(n, device=self.master[i].device)
            m, v, master = self.m[i], self.v[i], self.master[i]
            m.mul_(self.b1).add_(gv.to(m.dtype), alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(gv.to(v.dtype), gv.to(v.dtype), value=1 - self.b2)
            if self.wd:
                master.mul_(1 - self.lr * self.wd)
            denom = (v.float() / bc2).sqrt_().add_(self.eps)
            master.addcdiv_(m.float(), denom, value=-self.lr / bc1)
        for i, ci in enumerate(self.owned):
            pi, off, n, _ = self.chunks[ci]
            self._view(pi, off, n).copy_(self.master[i])
        if self.world > 1:
            for ci, (pi, off, n, owner) in enumerate(self.chunks):
                dist.broadcast(self._view(pi, off, n), src=owner)

    def state_dict(self):
        return {'tag': self.tag, 't': self.t, 'world': self.world,
                'layout': [(c[0], c[1], c[2]) for i, c in enumerate(self.chunks) if c[3] == self.rank],
                'master': self.master, 'm': self.m, 'v': self.v}

    def load_state_dict(self, sd):
        if sd.get('tag') != self.tag:
            raise ValueError('optimizer state has a different format (v1 used ZeroRedundancyOptimizer)')
        if int(sd.get('world', self.world)) != self.world:
            raise ValueError(f"optimizer shards were written with world_size={sd.get('world')}, now {self.world}")
        layout = [(c[0], c[1], c[2]) for i, c in enumerate(self.chunks) if c[3] == self.rank]
        if [tuple(x) for x in sd['layout']] != layout:
            raise ValueError('optimizer shard layout does not match the current parameter list')
        self.t = int(sd['t'])
        for i in range(len(self.master)):
            self.master[i].copy_(sd['master'][i].to(self.master[i].device))
            self.m[i].copy_(sd['m'][i].to(self.m[i].device))
            self.v[i].copy_(sd['v'][i].to(self.v[i].device))

    def save(self, out, rank):
        tmp = out / f'.optimizer_rank_{rank}.pt.tmp'
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, out / f'optimizer_rank_{rank}.pt')


class PlainAdamW:
    """Fallback: plain replicated AdamW.  Correct but not memory-optimal, and at
    lr<=1e-6 with bf16 parameters the update is rounded away (see above)."""

    tag = 'adamw-v1'

    def __init__(self, params, world=1, rank=0, lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.opt = torch.optim.AdamW(list(params), lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, foreach=False)

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt.step()

    def state_dict(self):
        return {'tag': self.tag, 'state': self.opt.state_dict()}

    def load_state_dict(self, sd):
        if sd.get('tag') != self.tag:
            raise ValueError('optimizer state has a different format')
        self.opt.load_state_dict(sd['state'])

    def save(self, out, rank):
        tmp = out / f'.optimizer_rank_{rank}.pt.tmp'
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, out / f'optimizer_rank_{rank}.pt')


def build_probe(params, k=6):
    """CPU snapshots of a few mid-sized tensors, used to detect frozen weights."""
    cand = [(i, p) for i, p in enumerate(params) if 1 << 20 <= p.numel() <= 1 << 26]
    if not cand:
        cand = list(enumerate(params))[:k]
    picked = cand[::max(1, len(cand) // k)][:k]
    return [(i, p.detach().to('cpu', copy=True)) for i, p in picked]


def probe_change_frac(probe, params):
    tot = ch = 0
    for i, snap in probe:
        cur = params[i].detach().to('cpu')
        tot += snap.numel()
        ch += int((cur != snap).sum())
    return ch / max(tot, 1)


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def fit_activation_budget(dev, static_bytes, pack_tokens, kl_chunk, vocab, hidden, layers,
                          reserve_gb=1.0, pack_share=0.62, chunk_share=0.28):
    """Cap --pack-tokens / --kl-chunk-tokens so the training step fits in VRAM.

    Gradient checkpointing keeps one input tensor per transformer block, i.e.
    2 * hidden * layers bytes per padded token, and one KL chunk holds ~5 fp32
    [chunk, vocab] temporaries.  Measured on the first smoke run: 16384 pack
    tokens + 1024 chunk tokens + 23.4 GiB of static state overflowed 31.36 GiB.
    The sampling pass and the training pass do not overlap, so their peaks are
    budgeted separately: this function caps the training side, and the KV cache
    of a rollout span is reported next to it in the `memory_budget` event.
    """
    total = torch.cuda.get_device_properties(dev).total_memory
    budget = max(0, total - static_bytes - int(reserve_gb * 2 ** 30))
    per_pack = max(2 * hidden * layers, 1)
    per_chunk = max(5 * vocab * 4, 1)
    max_pack = int(budget * pack_share / per_pack)
    max_chunk = int(budget * chunk_share / per_chunk)
    pack = 0 if pack_tokens <= 0 else min(pack_tokens, max(1, max_pack))
    chunk = max(64, min(kl_chunk, max(1, max_chunk)))
    return pack, chunk, budget, max_pack, max_chunk


def row_prompts(tok, row, a):
    """Student prompt, teacher prompt and rollout budget.

    The two prompts share the user turn and end in the same assistant header; the
    reference solution lives in the teacher's system message.  The shared tail is
    asserted, because a teacher prompt that ends in teacher-only text teaches the
    student to continue that text (the v2 marker leak).
    """
    user = {'role': 'user', 'content': row['question']}
    sp = templ(tok, [{'role': 'system', 'content': SYSTEM}, user])
    tp = templ(tok, [{'role': 'system', 'content': privileged_system(row['selected_completion'])}, user])
    if len(sp) >= 32 and len(tp) >= 32 and sp[-32:] != tp[-32:]:
        raise ValueError('teacher prompt does not share the student prompt tail; the KL would '
                         'train the student to continue teacher-only scaffolding')
    return sp, tp, min(a.max_length - len(sp), a.max_new_tokens)


def epoch_order(rows, seed):
    """The epoch's sample order: a plain seeded shuffle, as in v1.

    Sorting rows by reference length before batching was measured on
    shared_core_len10240_matched.jsonl and buys 1.3% of padded teacher tokens
    (1.036 -> 1.050) and nothing at all on the student side, whose length is set
    by the rollout rather than by the prompt.  That is not worth deviating from
    the data order of the RFT arm, so the sampler keeps the plain shuffle and
    only `pack_rows` groups by length within an update.
    """
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return order


def generate_span(student, tok, rows, start, dev, a, tag):
    """Sample the rollouts of a whole span of updates in one batched pass.

    Returns ({index in `local`: (student_seq, teacher_seq, gen_len)}, stats).
    Dropped rows map to nothing.  One decode step costs the same for 4 rows and
    for 12 (measured: 26.7 vs 26.8 ms; a step is dominated by per-kernel
    overhead, not by the batch), so covering several updates in one call is the
    cheapest way to raise GPU utilisation: the step count is fixed by the
    longest row of the batch either way.
    """
    prompts, tprompts, rooms = [], [], []
    for row in rows:
        sp, tp, room = row_prompts(tok, row, a)
        prompts.append(sp)
        tprompts.append(tp)
        rooms.append(room)
    keep = [i for i in range(len(rows)) if rooms[i] > 0]
    stats = {'rows': len(rows), 'skipped': len(rows) - len(keep), 'truncated': 0, 'trimmed': 0,
             'tokens': [], 'seconds': 0.0, 'context_tokens': [len(x) for x in prompts]}
    out = {}
    if not keep:
        return out, stats
    gin, gmask, gwidth = pad_left([prompts[i] for i in keep], tok.pad_token_id, dev)
    budget = max(rooms[i] for i in keep)
    phase = time.monotonic()
    student.config.use_cache = True
    student.eval()
    try:
        with torch.no_grad():
            gout = student.generate(gin, attention_mask=gmask, do_sample=True,
                                    temperature=a.temperature, top_p=a.top_p,
                                    max_new_tokens=budget, pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id, use_cache=True)
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({'event': 'oom', 'phase': 'generate', 'rollout_buffer': a.rollout_buffer,
                          'rows': len(rows), 'prompt_tokens': [len(x) for x in prompts],
                          'generation_budget': budget, **tag}, ensure_ascii=False), flush=True)
        raise
    finally:
        student.config.use_cache = False
        student.train()
    stats['seconds'] = time.monotonic() - phase
    samples = gout[:, gwidth:].tolist()
    del gout, gin, gmask
    for k, i in enumerate(keep):
        raw = samples[k]
        g = trim(raw, tok.eos_token_id)[:rooms[i]]
        stats['truncated'] += int(bool(g) and tok.eos_token_id is not None and tok.eos_token_id not in raw)
        if not g:
            stats['skipped'] += 1
            continue
        tp = tprompts[i]
        if len(tp) + len(g) > a.teacher_max_length:
            tp = fit_tokens(tp, max(1, a.teacher_max_length - len(g)))
            stats['trimmed'] += 1
        out[start + i] = (prompts[i] + g, tp + g, len(g))
        stats['tokens'].append(len(g))
    return out, stats


def save_ckpt(student, opt, out, rank, state):
    out.mkdir(parents=True, exist_ok=True)
    opt.save(out, rank)
    if rank == 0:
        tmp = out / '.student.pt.tmp'
        torch.save(student.state_dict(), tmp)
        os.replace(tmp, out / 'student.pt')
        ts = out / '.trainer_state.json.tmp'
        ts.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        os.replace(ts, out / 'trainer_state.json')
    dist.barrier()


def main():
    global MODEL, SYSTEM
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL, help='coldstart HF directory or model ID')
    ap.add_argument('--system-prompt', default=str(ROOT / 'prompts/system.txt'))
    ap.add_argument('--shared', default=str(ROOT / 'data/train/shared_core_len8192.jsonl'))
    ap.add_argument('--output-dir', default=str(ROOT / 'outputs/opsd'))
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-new-tokens', type=int, default=8192)
    ap.add_argument('--max-length', type=int, default=10240, help='student sequence cap')
    ap.add_argument('--teacher-max-length', type=int, default=20480, help='teacher sequence cap')
    ap.add_argument('--micro-batch-size', type=int, default=4)
    ap.add_argument('--global-batch-size', type=int, default=8)
    ap.add_argument('--grad-accum-steps', type=int, default=0, help='0 = derive from world/micro/global')
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=.95)
    ap.add_argument('--kl-clip', type=float, default=5.0, help='soft saturating cap; 0 disables')
    ap.add_argument('--kl-chunk-tokens', type=int, default=1024)
    ap.add_argument('--pack-tokens', type=int, default=16384, help='max padded tokens per backbone forward (0 = one sequence)')
    ap.add_argument('--rollout-buffer', type=int, default=1,
                    help='sample this many updates per generation call (the decode step count is set by the '
                         'longest row either way, so >1 is nearly free and uses the GPU much better)')
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--student-dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    ap.add_argument('--teacher-device', choices=['cuda', 'cpu'], default=os.environ.get('TEACHER_DEVICE', 'cuda'))
    ap.add_argument('--optimizer', choices=['master', 'adamw'], default='master')
    ap.add_argument('--moment-dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    ap.add_argument('--master-dtype', choices=['float32', 'bfloat16'], default='float32')
    ap.add_argument('--max-batches', type=int, default=0, help='debug limit; 0 means all')
    ap.add_argument('--warn-frozen-after', type=int, default=120, help='warn if no bf16 weight moved after N updates')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--resume-model-only', action='store_true', help='load student + progress but start with a fresh optimizer')
    a = ap.parse_args()
    if not a.model:
        ap.error('Set --model or COLDSTART_MODEL to the actual coldstart checkpoint')
    MODEL = a.model
    SYSTEM = Path(a.system_prompt).read_text(encoding='utf-8').strip()

    rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    # Build the NCCL communicator now, while the torchrun store is certainly
    # alive: it is built lazily on the first collective, and if the launcher is
    # gone by then (e.g. a detached `nohup` job whose parent exited) the
    # rendezvous fails with "store->get('0') got error: Broken pipe".
    dist.all_reduce(torch.zeros(1, device=dev))
    world = dist.get_world_size()
    set_seed(a.seed + rank)

    accum = a.grad_accum_steps or max(1, a.global_batch_size // (world * a.micro_batch_size))
    expected = world * a.micro_batch_size * accum
    if expected != a.global_batch_size:
        raise ValueError(f'global batch mismatch: world*micro*accum={expected}, expected {a.global_batch_size}')

    allrows = [json.loads(x) for x in Path(a.shared).read_text(encoding='utf-8').splitlines() if x.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.padding_side = 'left'
    tok.pad_token = tok.pad_token or tok.eos_token

    sdtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[a.student_dtype]
    student = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=sdtype,
                                                   attn_implementation='sdpa').to(dev)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    s_backbone, s_head = student.model, student.lm_head

    teacher_dev = torch.device('cpu') if a.teacher_device == 'cpu' else dev
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation='sdpa').to(teacher_dev).eval()
    teacher.requires_grad_(False)
    t_backbone = teacher.model
    t_head = teacher.lm_head if teacher_dev == dev else teacher.lm_head.to(dev)

    params = [p for p in student.parameters() if p.requires_grad]
    mdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[a.master_dtype]
    motype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[a.moment_dtype]
    if a.optimizer == 'master':
        opt = ShardedMasterAdamW(params, world, rank, lr=a.lr, weight_decay=0.01,
                                 master_dtype=mdtype, moment_dtype=motype)
    else:
        opt = PlainAdamW(params, lr=a.lr, weight_decay=0.01)

    out = Path(a.output_dir)
    statepath = out / 'trainer_state.json'
    start_epoch = update = resume_batches = 0
    if a.resume and statepath.exists() and (out / 'student.pt').exists():
        student.load_state_dict(torch.load(out / 'student.pt', map_location='cpu'), strict=True)
        opt.sync_master_from_params() if hasattr(opt, 'sync_master_from_params') else None
        saved = json.loads(statepath.read_text())
        start_epoch = int(saved.get('epoch', 0))
        update = int(saved.get('update_step', 0))
        resume_batches = int(saved.get('epoch_batches', -1))
        shard = out / f'optimizer_rank_{rank}.pt'
        if a.resume_model_only or saved.get('optimizer_format') != opt.tag:
            print(json.dumps({'event': 'resume', 'optimizer_state': 'reinitialised',
                              'previous_format': saved.get('optimizer_format')}, ensure_ascii=False), flush=True)
        elif shard.exists():
            try:
                opt.load_state_dict(torch.load(shard, map_location='cpu'))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({'event': 'resume', 'optimizer_state': 'load_failed', 'error': str(exc)},
                                 ensure_ascii=False), flush=True)
        dist.barrier()

    probe = build_probe(params) if rank == 0 else []
    static_bytes = (tensor_bytes(student.parameters()) + tensor_bytes(teacher.parameters())
                    + tensor_bytes(getattr(opt, 'master', [])) + tensor_bytes(getattr(opt, 'm', []))
                    + tensor_bytes(getattr(opt, 'v', [])) + tensor_bytes(params))
    pack_tokens, kl_chunk, act_budget, max_pack, max_chunk = fit_activation_budget(
        dev, static_bytes, a.pack_tokens, a.kl_chunk_tokens, student.config.vocab_size,
        student.config.hidden_size, student.config.num_hidden_layers)
    if rank == 0:
        print(json.dumps({
            'event': 'config', 'world': world, 'micro_batch_size': a.micro_batch_size,
            'grad_accum_steps': accum, 'global_batch_size': a.global_batch_size,
            'student_dtype': a.student_dtype, 'optimizer': opt.tag, 'moment_dtype': a.moment_dtype,
            'teacher_device': str(teacher_dev), 'rollout_buffer': a.rollout_buffer,
            'rollout_rows_per_rank': a.micro_batch_size * accum * max(1, a.rollout_buffer),
            'max_length': a.max_length,
            'teacher_max_length': a.teacher_max_length, 'max_new_tokens': a.max_new_tokens,
            'lr': a.lr, 'kl_clip': a.kl_clip, 'rows': len(allrows),
            'mem_allocated_gb': round(torch.cuda.memory_allocated(dev) / 2 ** 30, 2),
        }, ensure_ascii=False), flush=True)
        kv_per_token = (2 * student.config.num_key_value_heads
                        * (student.config.hidden_size // student.config.num_attention_heads)
                        * 2 * student.config.num_hidden_layers)
        kv_max = (a.micro_batch_size * accum * max(1, a.rollout_buffer) * a.max_length * kv_per_token)
        kv_expected = (a.micro_batch_size * accum * max(1, a.rollout_buffer)
                       * (a.max_new_tokens + 2048) * kv_per_token)
        print(json.dumps({
            'event': 'memory_budget',
            'static_gb': round(static_bytes / 2 ** 30, 2),
            'device_total_gb': round(torch.cuda.get_device_properties(dev).total_memory / 2 ** 30, 2),
            'activation_budget_gb': round(act_budget / 2 ** 30, 2),
            'pack_tokens': pack_tokens, 'pack_tokens_cap': max_pack,
            'kl_chunk_tokens': kl_chunk, 'kl_chunk_tokens_cap': max_chunk,
            'train_peak_est_gb': round((static_bytes + pack_tokens * 2 * student.config.hidden_size
                                        * student.config.num_hidden_layers
                                        + kl_chunk * 5 * student.config.vocab_size * 4) / 2 ** 30, 2),
            'kv_per_token_kb': round(kv_per_token / 1024, 1),
            'kv_span_expected_gb': round(kv_expected / 2 ** 30, 2),
            'kv_span_max_gb': round(kv_max / 2 ** 30, 2),
            'free_gb': round((torch.cuda.get_device_properties(dev).total_memory - static_bytes) / 2 ** 30, 2),
        }, ensure_ascii=False), flush=True)

    chunk_kl = make_chunk_kl(s_head, t_head, a.kl_clip)
    local_tokens = local_skipped = local_truncated = local_trimmed = local_samples = local_rolled = 0
    started = time.monotonic()
    step = a.micro_batch_size * accum          # rows per rank and update
    buffer = max(1, a.rollout_buffer)
    span = step * buffer                       # rows per rank and sampling pass
    unit = world * step
    batches_per_epoch = (len(allrows) - (len(allrows) % unit)) // unit

    for epoch in range(start_epoch, a.epochs):
        order = epoch_order(allrows, a.seed + epoch)
        keep_n = len(order) - (len(order) % unit)
        dropped = len(order) - keep_n
        local = [allrows[i] for i in order[:keep_n]][rank::world]
        # resume: skip exactly the updates this epoch already consumed.  The old
        # `update % batches_per_epoch` replay of a whole epoch is gone, and
        # `epoch_batches` is saved explicitly so the offset stays exact even if an
        # update was skipped for lack of valid rollouts.
        if epoch == start_epoch:
            skip_batches = (resume_batches if resume_batches >= 0
                            else max(0, update // buffer - epoch * batches_per_epoch))
        else:
            skip_batches = 0
        epoch_batches = skip_batches
        opt.zero_grad(set_to_none=True)
        roll_cache = {}
        for batch_no, base_i in enumerate(range(0, len(local), step)):
            if batch_no < skip_batches:
                continue
            if a.max_batches and (batch_no - skip_batches) >= a.max_batches:
                break
            epoch_batches = batch_no + 1
            # Rollouts are sampled once per span of `buffer` updates and then
            # consumed update by update.  The decode step count is set by the
            # longest row of the batch either way, so the extra rows of a span
            # cost almost no time (only their KV cache costs memory).  Resuming
            # mid-span regenerates the span, hence the cache keyed on its start.
            span_start = base_i - (batch_no % buffer) * step
            if span_start not in roll_cache:
                roll_cache.clear()
                roll, rstats = generate_span(student, tok, local[span_start:span_start + span],
                                             span_start, dev, a,
                                             {'epoch': epoch + 1, 'update_step': update + 1})
                roll_cache[span_start] = roll
                local_skipped += rstats['skipped']
                local_rolled += rstats['rows']
                local_truncated += rstats['truncated']
                local_trimmed += rstats['trimmed']
                if rank == 0:
                    print(json.dumps({'event': 'span', 'epoch': epoch + 1, 'update_step': update + 1,
                                      'rows': rstats['rows'], 'generated_tokens': rstats['tokens'],
                                      'gen_seconds': round(rstats['seconds'], 1)}, ensure_ascii=False), flush=True)
                    if not roll:
                        print(json.dumps({'event': 'batch_skipped', 'epoch': epoch + 1, 'update_step': update + 1,
                                          'reason': 'no_context_room', 'count': rstats['rows'],
                                          'prompt_tokens': rstats['context_tokens']}, ensure_ascii=False), flush=True)
            roll = roll_cache[span_start]
            accum_valid = accum_samples = accum_clipped = 0
            accum_loss = 0.0
            train_seconds = 0.0
            for ai in range(accum):
                lo = base_i + ai * a.micro_batch_size
                batch = [x for x in (roll.get(i) for i in range(lo, min(lo + a.micro_batch_size, len(local))))
                         if x is not None]
                if not batch:
                    continue
                sseqs = [x[0] for x in batch]
                tseqs = [x[1] for x in batch]
                genlens = [x[2] for x in batch]
                phase1 = time.monotonic()

                ht = [None] * len(sseqs)
                for pb in pack_rows([len(t) for t in tseqs], pack_tokens):
                    got = hidden_states(t_backbone, [tseqs[i] for i in pb], teacher_dev, grad=False)
                    for j, i in enumerate(pb):
                        ht[i] = got[j]
                del got

                # Forward + backward one packed group at a time: the
                # gradient-checkpoint activations of a group (~2*hidden*layers
                # bytes per padded token) are released before the next group is
                # forwarded, so the peak is one group, not the whole update.
                # The rows of a group share one backbone graph, so they must be
                # summed into a single scalar and backpropagated together; a
                # per-row backward would free that graph after the first row.
                for pb in pack_rows([len(s) for s in sseqs], pack_tokens):
                    hs = hidden_states(s_backbone, [sseqs[i] for i in pb], dev, grad=True)
                    group = None
                    for j, i in enumerate(pb):
                        ss = len(sseqs[i]) - genlens[i]
                        ts = len(tseqs[i]) - genlens[i]
                        gs = genlens[i]
                        total, raw_kl, nclip = None, 0.0, 0
                        for begin in range(0, gs, kl_chunk):
                            end = min(begin + kl_chunk, gs)
                            hs_c = hs[j][ss - 1 + begin: ss - 1 + end]
                            ht_c = ht[i][ts - 1 + begin: ts - 1 + end]
                            assert end - begin == hs_c.shape[0] == ht_c.shape[0], 'chunk alignment'
                            l, r, c = ckpt(chunk_kl, hs_c, ht_c, use_reentrant=False, determinism_check='none')
                            total = l if total is None else total + l
                            raw_kl += float(r)
                            nclip += int(c)
                        term = total / gs
                        group = term if group is None else group + term
                        accum_valid += gs
                        accum_samples += 1
                        accum_loss += raw_kl / gs
                        accum_clipped += nclip
                    group.backward()
                    for i in pb:
                        ht[i] = None
                    del hs, group
                train_seconds += time.monotonic() - phase1

            n_valid = gsum(accum_samples, dev)
            if n_valid > 0:
                sync_gradients(params, n_valid)
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
                opt.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                local_tokens += accum_valid
                local_samples += accum_samples
                # every counter below is reduced on all ranks, only rank 0 prints
                g_loss = gsum(accum_loss, dev)
                g_clip = gsum(accum_clipped, dev)
                g_tokens = gsum(local_tokens, dev)
                g_skip = gsum(local_skipped, dev)
                g_trunc = gsum(local_truncated, dev)
                g_trim = gsum(local_trimmed, dev)
                g_samples = gsum(local_samples, dev)
                g_rolled = gsum(local_rolled, dev)
                if rank == 0:
                    changed = probe_change_frac(probe, params)
                    print(json.dumps({'epoch': epoch + 1, 'update_step': update,
                                      'samples': int(n_valid), 'loss': round(g_loss / max(n_valid, 1), 6),
                                      'grad_norm': round(gnorm, 5), 'param_changed_frac': round(changed, 5),
                                      'kl_clipped_tokens': int(g_clip),
                                      'consumed_tokens': int(g_tokens), 'skipped': int(g_skip),
                                      'truncated': int(g_trunc),
                                      'valid_samples': int(g_samples), 'rolled_rows': int(g_rolled),
                                      'truncated_pct': round(100 * g_trunc / max(g_rolled, 1), 2),
                                      'teacher_trimmed': int(g_trim), 'dropped_per_epoch': dropped,
                                      'peak_mem_gb': round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
                                      'train_seconds': round(train_seconds, 1),
                                      'gpu_seconds': round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
                    if changed == 0.0 and update == a.warn_frozen_after:
                        print(json.dumps({'event': 'warning', 'update_step': update,
                                          'message': 'no bf16 weight changed yet; with bf16 parameters a small lr is '
                                                     'rounded away - raise --lr or keep --optimizer master'}, ensure_ascii=False), flush=True)
                if update % a.save_every == 0:
                    save_ckpt(student, opt, out, rank, {
                        'epoch': epoch, 'update_step': update, 'epoch_batches': epoch_batches,
                        'consumed_tokens': int(g_tokens),
                        'skipped': int(g_skip),
                        'truncated': int(g_trunc),
                        'valid_samples': int(g_samples),
                        'rolled_rows': int(g_rolled),
                        'truncated_pct': round(100 * g_trunc / max(g_rolled, 1), 2),
                        'gpu_seconds': time.monotonic() - started,
                        'global_batch_size': a.global_batch_size, 'micro_batch_size': a.micro_batch_size,
                        'grad_accum_steps': accum, 'max_length': a.max_length,
                        'max_new_tokens': a.max_new_tokens, 'world_size': world,
                        'optimizer_format': opt.tag, 'complete': False})

    g_tokens = int(gsum(local_tokens, dev))
    g_skip = int(gsum(local_skipped, dev))
    g_trunc = int(gsum(local_truncated, dev))
    g_samples = int(gsum(local_samples, dev))
    g_rolled = int(gsum(local_rolled, dev))
    dist.barrier()
    save_ckpt(student, opt, out, rank, {
        'epoch': a.epochs, 'update_step': update, 'epoch_batches': 0,
        'consumed_tokens': g_tokens,
        'skipped': g_skip,
        'truncated': g_trunc,
        'valid_samples': g_samples,
        'rolled_rows': g_rolled,
        'truncated_pct': round(100 * g_trunc / max(g_rolled, 1), 2),
        'gpu_seconds': time.monotonic() - started,
        'global_batch_size': a.global_batch_size, 'micro_batch_size': a.micro_batch_size,
        'grad_accum_steps': accum, 'max_length': a.max_length,
        'max_new_tokens': a.max_new_tokens, 'world_size': world,
        'optimizer_format': opt.tag, 'complete': True})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
