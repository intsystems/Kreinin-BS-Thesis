"""Ярус C2: pretraining GPT-2 124M (nanoGPT-стиль) с AdamL2/AdamW/AdamWH и послойной диагностикой.

Запуск (одна карта): python -m theory_bench.gpt_bench --mode w --lr 6e-4 --wd 0.1 --tokens 1e8 --tag pilot
Результат: results/gpt/<tag>_<mode>_lr<lr>_wd<wd>_s<seed>.json
Диагностика каждые --diag-every шагов на фиксированном пробном батче: послойно ||grad f||^2,
||grad F||^2, ||grad F_tilde||^2 (D̂ = sqrt(v̂)+eps — ровно как в шаге), медианы D̂.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .dl_bench import AdamThreeModes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "fineweb")
RES = os.path.join(HERE, "results", "gpt")
os.makedirs(RES, exist_ok=True)


# ---------------------------------------------------------------- модель (GPT-2 124M)
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp1 = nn.Linear(d, 4 * d)
        self.mlp2 = nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        z = self.ln2(x)
        x = x + self.mlp2(Fn.gelu(self.mlp1(z)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab=50304, ctx=1024, d=768, n_layer=12, n_head=12):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.ctx = ctx
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = Fn.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------- данные
class BinData:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def batch(self, bs, ctx, device, rng):
        ix = rng.integers(0, len(self.data) - ctx - 1, size=bs)
        x = torch.stack([torch.from_numpy(self.data[i:i + ctx].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def layer_group(name):
    name = name.removeprefix("_orig_mod.")  # префикс torch.compile
    if name.startswith("wte") or name.startswith("wpe"):
        return "emb"
    if name.startswith("blocks."):
        li = int(name.split(".")[1])
        part = "attn" if (".attn" in name or ".proj" in name) else ("mlp" if ".mlp" in name else "ln")
        third = "low" if li < 4 else ("mid" if li < 8 else "high")
        return f"{third}_{part}"
    return "final"


def probe_diag(model, opt, probe_xy, wd):
    model.zero_grad(set_to_none=True)
    x, y = probe_xy
    _, loss = model(x, y)
    loss.backward()
    stats = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            D = opt.d_hat(p)
            if D is None:
                continue
            gf = p.grad
            gF = gf + wd * p
            gFt = gf + wd * D * p
            g = layer_group(name)
            s = stats.setdefault(g, {"gf2": 0.0, "gF2": 0.0, "gFt2": 0.0, "dsum": 0.0, "n": 0})
            s["gf2"] += float((gf * gf).sum())
            s["gF2"] += float((gF * gF).sum())
            s["gFt2"] += float((gFt * gFt).sum())
            s["dsum"] += float(D.sum())
            s["n"] += D.numel()
    for s in stats.values():
        s["d_mean"] = s["dsum"] / max(s["n"], 1)
        del s["dsum"]
    glob = {k: sum(s[k] for s in stats.values()) for k in ("gf2", "gF2", "gFt2")}
    model.zero_grad(set_to_none=True)
    return {"layers": stats, "global": glob}


def train(mode, lr, wd, tokens, seed, device, tag, micro_bs=8, accum=30,
          beta2=0.95, diag_every=200, val_every=200, ckpt_every=500, eps=1e-8):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    raw = GPT().to(device)
    model = torch.compile(raw)
    opt = AdamThreeModes(model.parameters(), lr=lr, wd=wd, mode=mode, betas=(0.9, beta2), eps=eps)
    tr = BinData(os.path.join(DATA, "train.bin"))
    va = BinData(os.path.join(DATA, "val.bin"))
    rng = np.random.default_rng(seed)
    tokens_per_step = micro_bs * 1024 * accum
    total_steps = int(tokens / tokens_per_step)
    warmup = min(300, total_steps // 20)
    ckpt_path = os.path.join(RES, f"ckpt_{tag}_{mode}_lr{lr:g}_wd{wd:g}_s{seed}.pt")
    # eps учитывается в имени результата через tag

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))  # cosine до 0.1*lr

    probe_xy = va.batch(8, 1024, device, np.random.default_rng(777))
    hist = {"mode": mode, "lr": lr, "wd": wd, "seed": seed, "tokens": tokens,
            "beta2": beta2, "tokens_per_step": tokens_per_step,
            "val_loss": [], "val_at_tokens": [], "diag": {}}
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        hist = ck["hist"]
        start_step = ck["step"] + 1
        print(f"[{tag} {mode}] resumed from step {start_step}", flush=True)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "hist": hist, "step": step}, tmp)
        os.replace(tmp, ckpt_path)

    t0 = time.time()
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = tr.batch(micro_bs, 1024, device, rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / accum).backward()
        opt.step()
        if step % val_every == 0 or step == total_steps - 1:
            model.eval()
            with torch.no_grad():
                vls = []
                vrng = np.random.default_rng(123)
                for _ in range(8):
                    x, y = va.batch(8, 1024, device, vrng)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, vl = model(x, y)
                    vls.append(float(vl))
            model.train()
            hist["val_loss"].append(float(np.mean(vls)))
            hist["val_at_tokens"].append((step + 1) * tokens_per_step)
            tps = (step + 1) * tokens_per_step / max(time.time() - t0, 1e-9)
            print(f"[{tag} {mode} lr={lr:g} wd={wd:g}] step {step}/{total_steps} "
                  f"val={hist['val_loss'][-1]:.4f} ({tps/1e3:.0f}k tok/s)", flush=True)
        if step % diag_every == 0 or step == total_steps - 1:
            hist["diag"][str(step)] = probe_diag(model, opt, probe_xy, wd)
        if ckpt_every and (step % ckpt_every == 0 and step > 0 or step == total_steps - 1):
            save_ckpt(step)
    hist["final_val"] = hist["val_loss"][-1]
    hist["wall_s"] = time.time() - t0
    fname = f"{tag}_{mode}_lr{lr:g}_wd{wd:g}_s{seed}.json"
    with open(os.path.join(RES, fname), "w") as f:
        json.dump(hist, f)
    print("saved", fname, "final_val", hist["final_val"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["l2", "w", "wh"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="c2")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=30)
    ap.add_argument("--eps", type=float, default=1e-8)
    args = ap.parse_args()
    train(args.mode, args.lr, args.wd, args.tokens, args.seed, args.device, args.tag,
          micro_bs=args.micro_bs, accum=args.accum, eps=args.eps)
