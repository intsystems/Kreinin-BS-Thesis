"""C2-restarts: ветвление обучения GPT-2 с сохранённого чекпойнта с другим сидом данных.

Показывает, что поведение критериев ||grad F|| / ||grad F_tilde|| устойчиво к рестартам:
все ветки из одной точки с разной стохастикой дают один и тот же качественный эффект.

Запуск: python -m theory_bench.restart_branch --mode w --branch-seed 101 --steps 600 --device cuda:0
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .gpt_bench import GPT, BinData, probe_diag, DATA, RES
from .dl_bench import AdamThreeModes


def main(mode, branch_seed, steps, device, micro_bs=8, accum=30, lr_frac=0.1):
    ck = torch.load(os.path.join(RES, f"ckpt_main_{mode}_lr0.0003_wd0.01_s0.pt"),
                    map_location=device, weights_only=False)
    raw = GPT().to(device)
    raw.load_state_dict(ck["model"])
    model = torch.compile(raw)
    opt = AdamThreeModes(model.parameters(), lr=3e-4 * lr_frac, wd=0.01, mode=mode,
                         betas=(0.9, 0.95))
    opt.load_state_dict(ck["opt"])
    for g in opt.param_groups:
        g["lr"] = 3e-4 * lr_frac  # продолжение на финальном уровне косинуса
    tr = BinData(os.path.join(DATA, "train.bin"))
    va = BinData(os.path.join(DATA, "val.bin"))
    rng = np.random.default_rng(branch_seed)  # ДРУГОЙ поток данных
    probe_xy = va.batch(8, 1024, device, np.random.default_rng(777))
    wd = 0.01
    hist = {"mode": mode, "branch_seed": branch_seed, "from_step": ck["step"],
            "steps": steps, "val_loss": [], "diag": {}}
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = tr.batch(micro_bs, 1024, device, rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / accum).backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vls = []
                vrng = np.random.default_rng(123)
                for _ in range(4):
                    x, y = va.batch(8, 1024, device, vrng)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, vl = model(x, y)
                    vls.append(float(vl))
            model.train()
            hist["val_loss"].append(float(np.mean(vls)))
            hist["diag"][str(step)] = probe_diag(model, opt, probe_xy, wd)
            print(f"[branch {mode} seed{branch_seed}] {step}/{steps} val={hist['val_loss'][-1]:.4f} "
                  f"({(step+1)*micro_bs*1024*accum/max(time.time()-t0,1e-9)/1e3:.0f}k tok/s)", flush=True)
    fname = f"branch_{mode}_seed{branch_seed}.json"
    with open(os.path.join(RES, fname), "w") as f:
        json.dump(hist, f)
    print("saved", fname, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="w", choices=["l2", "w"])
    ap.add_argument("--branch-seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    main(args.mode, args.branch_seed, args.steps, args.device)
