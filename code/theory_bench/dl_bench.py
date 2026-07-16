"""Ярус B: ResNet-18 / CIFAR-10, сравнение AdamL2 / AdamW / AdamWH (B1: heatmap eta x lambda).

Один прогон:
  python -m theory_bench.dl_bench --mode w --lr 1e-3 --wd 1e-2 --seed 0 --epochs 50 --device cuda:0

Результат: results/dl/<tag>_<mode>_lr<lr>_wd<wd>_s<seed>.json
  (кривые train loss / test acc + послойная диагностика ||grad F||, ||grad F_tilde||, статистики D̂).
Диагностика использует ровно ту D̂ = sqrt(v̂) + eps, что и шаг оптимизатора (фикс бага
оригинального code/optimizers.py, где g_adamw считался без bias-correction).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results", "dl")
os.makedirs(RES, exist_ok=True)


# ---------------------------------------------------------------- оптимизатор
class AdamThreeModes(torch.optim.Optimizer):
    """Adam с тремя режимами регуляризации: l2 (в градиент), w (decoupled), wh (масштабированный)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0, mode="w"):
        assert mode in ("l2", "w", "wh")
        defaults = dict(lr=lr, betas=betas, eps=eps, wd=wd, mode=mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd, mode = (group["lr"], group["betas"], group["eps"],
                                           group["wd"], group["mode"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                if mode == "l2":
                    g = g.add(p, alpha=wd)
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = st["m"] / (1 - b1 ** t)
                v_hat = st["v"] / (1 - b2 ** t)
                denom = v_hat.sqrt_().add_(eps)
                if mode == "wh":
                    p.addcdiv_(m_hat.add(p, alpha=wd), denom, value=-lr)
                elif mode == "w":
                    p.addcdiv_(m_hat, denom, value=-lr)
                    p.add_(p, alpha=-lr * wd)
                else:
                    p.addcdiv_(m_hat, denom, value=-lr)

    @torch.no_grad()
    def d_hat(self, p):
        """D̂ = sqrt(v̂) + eps — ровно как в шаге."""
        st = self.state.get(p)
        if not st:
            return None
        group = next(g for g in self.param_groups if any(q is p for q in g["params"]))
        v_hat = st["v"] / (1 - group["betas"][1] ** st["t"])
        return v_hat.sqrt() + group["eps"]


# ---------------------------------------------------------------- модель / данные
def resnet18_cifar(num_classes=10):
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class ViTTiny(nn.Module):
    """Компактный ViT для CIFAR: patch 4, dim 192, depth 9, heads 3, mean-pooling."""

    def __init__(self, num_classes=10, dim=192, depth=9, heads=3, patch=4):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_tok = (32 // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            blk = nn.ModuleDict({
                "ln1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
            })
            self.blocks.append(blk)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            h = b["ln1"](x)
            a, _ = b["attn"](h, h, h, need_weights=False)
            x = x + a
            x = x + b["mlp"](b["ln2"](x))
        return self.head(self.ln_f(x).mean(1))


def loaders(batch=128, workers=4, dataset="cifar10", randaug=False):
    if dataset == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        DS = torchvision.datasets.CIFAR10
    else:
        norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        DS = torchvision.datasets.CIFAR100
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if randaug:
        aug = [transforms.RandAugment(2, 9)] + aug
    tf_train = transforms.Compose(aug + [transforms.ToTensor(), norm])
    tf_test = transforms.Compose([transforms.ToTensor(), norm])
    tr = DS(DATA, train=True, download=True, transform=tf_train)
    te = DS(DATA, train=False, download=True, transform=tf_test)
    tr_plain = DS(DATA, train=True, download=False, transform=tf_test)
    lt = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,
                                     num_workers=workers, pin_memory=True, drop_last=True)
    lv = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    # фиксированный пробный батч (без аугментаций) для критериев
    g = torch.Generator().manual_seed(777)
    idx = torch.randperm(len(tr_plain), generator=g)[:2048]
    probe = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_plain, idx.tolist()),
                                        batch_size=512, shuffle=False, num_workers=2)
    return lt, lv, probe


# ---------------------------------------------------------------- диагностика
def layer_group(name):
    return name.split(".")[0]


@torch.no_grad()
def _noop():
    pass


def probe_diagnostics(model, opt, probe, wd, device):
    """Послойные ||grad f||^2, ||grad F||^2, ||grad F_tilde||^2 и статистики D̂
    на фиксированном пробном батче (f — чистый лосс без регуляризатора)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    n_tot = 0
    for x, y in probe:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = Fn.cross_entropy(out, y, reduction="sum")
        loss.backward()
        n_tot += y.numel()
    stats = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            gf = p.grad / n_tot
            D = opt.d_hat(p)
            if D is None:
                continue
            gF = gf + wd * p
            gFt = gf + wd * D * p
            gname = layer_group(name)
            s = stats.setdefault(gname, {"gf2": 0.0, "gF2": 0.0, "gFt2": 0.0,
                                         "d_sum": 0.0, "d_min": float("inf"),
                                         "d_max": 0.0, "n": 0})
            s["gf2"] += float((gf * gf).sum())
            s["gF2"] += float((gF * gF).sum())
            s["gFt2"] += float((gFt * gFt).sum())
            s["d_sum"] += float(D.sum())
            s["d_min"] = min(s["d_min"], float(D.min()))
            s["d_max"] = max(s["d_max"], float(D.max()))
            s["n"] += D.numel()
    for s in stats.values():
        s["d_mean"] = s["d_sum"] / max(s["n"], 1)
        del s["d_sum"]
    glob = {k: sum(s[k] for s in stats.values()) for k in ("gf2", "gF2", "gFt2")}
    model.zero_grad(set_to_none=True)
    model.train()
    return {"layers": stats, "global": glob}


# ---------------------------------------------------------------- обучение
def train(mode, lr, wd, seed, epochs, device, tag="b1", batch=128, diag_every=5,
          beta2=0.999, sched="cosine", dataset="cifar10", model_name="resnet18"):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    ncls = 10 if dataset == "cifar10" else 100
    if model_name == "resnet18":
        model = resnet18_cifar(ncls).to(device).to(memory_format=torch.channels_last)
        ls, randaug = 0.0, False
    else:
        model = ViTTiny(ncls).to(device)
        ls, randaug = 0.1, True
    if mode == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        opt = AdamThreeModes(model.parameters(), lr=lr, wd=wd, mode=mode, betas=(0.9, beta2))
    lt, lv, probe = loaders(batch=batch, dataset=dataset, randaug=randaug)
    steps_per_epoch = len(lt)
    warmup = 5 * steps_per_epoch
    total = epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        if sched == "const":
            return lr
        prog = (step - warmup) / max(total - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * prog))

    scaler = torch.amp.GradScaler("cuda")
    hist = {"train_loss": [], "test_acc": [], "diag": {}, "lr": lr, "wd": wd,
            "mode": mode, "seed": seed, "epochs": epochs, "beta2": beta2, "sched": sched,
            "dataset": dataset, "model": model_name}
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run_loss, nb = 0.0, 0
        for x, y in lt:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = Fn.cross_entropy(model(x), y, label_smoothing=ls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            opt.step()
            scaler.update()
            run_loss += float(loss.detach())
            nb += 1
            step += 1
        # test
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for x, y in lv:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                corr += int((out.argmax(1) == y).sum())
                tot += y.numel()
        hist["train_loss"].append(run_loss / max(nb, 1))
        hist["test_acc"].append(corr / tot)
        if (ep % diag_every == 0 or ep == epochs - 1) and mode != "sgd":
            hist["diag"][str(ep)] = probe_diagnostics(model, opt, probe, wd, device)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[{tag} {mode} lr={lr:g} wd={wd:g} s{seed}] ep{ep} "
                  f"loss={hist['train_loss'][-1]:.3f} acc={hist['test_acc'][-1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    hist["best_acc"] = max(hist["test_acc"])
    hist["final_acc"] = hist["test_acc"][-1]
    hist["wall_s"] = time.time() - t0
    fname = f"{tag}_{model_name}_{dataset}_{mode}_lr{lr:g}_wd{wd:g}_s{seed}.json" if tag.startswith("b4") \
        else f"{tag}_{mode}_lr{lr:g}_wd{wd:g}_s{seed}.json"
    with open(os.path.join(RES, fname), "w") as f:
        json.dump(hist, f)
    print("saved", fname, "best", hist["best_acc"], flush=True)
    return hist["best_acc"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["l2", "w", "wh", "sgd"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "const"])
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "vit"])
    args = ap.parse_args()
    train(args.mode, args.lr, args.wd, args.seed, args.epochs, args.device, args.tag,
          beta2=args.beta2, sched=args.sched, dataset=args.dataset, model_name=args.model)
