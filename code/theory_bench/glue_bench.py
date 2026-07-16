"""Ярус C1: fine-tuning RoBERTa-base на GLUE (RTE, MRPC, CoLA, STS-B) с AdamL2/AdamW/AdamWH.

Запуск: python -m theory_bench.glue_bench --task rte --mode w --lr 2e-5 --seed 0 --device cuda:0
Результат: results/glue/<task>_<mode>_lr<lr>_wd<wd>_s<seed>.json
Диагностика: медианы D̂ = sqrt(v̂)+eps по слоям энкодера (нижние vs верхние) в конце обучения.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from .dl_bench import AdamThreeModes

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "glue")
os.makedirs(RES, exist_ok=True)

TASKS = {
    "rte":  dict(keys=("sentence1", "sentence2"), n_labels=2, metric="acc", epochs=10, bs=16),
    "mrpc": dict(keys=("sentence1", "sentence2"), n_labels=2, metric="f1", epochs=10, bs=16),
    "cola": dict(keys=("sentence", None), n_labels=2, metric="mcc", epochs=10, bs=32),
    "stsb": dict(keys=("sentence1", "sentence2"), n_labels=1, metric="spearman", epochs=10, bs=16),
    "sst2": dict(keys=("sentence", None), n_labels=2, metric="acc", epochs=3, bs=32),
}


def metric_value(name, y_true, y_pred):
    if name == "acc":
        return float((np.array(y_true) == np.array(y_pred)).mean())
    if name == "f1":
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred))
    if name == "mcc":
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true, y_pred))
    if name == "spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(y_true, y_pred).statistic)
    raise ValueError(name)


def make_loader(ds, tok, keys, bs, shuffle, seed=0):
    k1, k2 = keys

    def collate(batch):
        texts1 = [b[k1] for b in batch]
        texts2 = [b[k2] for b in batch] if k2 else None
        enc = tok(texts1, texts2, truncation=True, max_length=128, padding=True,
                  return_tensors="pt")
        labels = torch.tensor([b["label"] for b in batch])
        return enc, labels

    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g,
                                       collate_fn=collate, num_workers=2)


def layer_dhat_stats(model, opt):
    """Медианы D̂ по слоям энкодера: инд. слоя -> медиана sqrt(v̂)+eps."""
    per_layer = {}
    for name, p in model.named_parameters():
        D = opt.d_hat(p)
        if D is None:
            continue
        if ".layer." in name:
            li = int(name.split(".layer.")[1].split(".")[0])
            key = f"layer{li:02d}"
        elif "embeddings" in name:
            key = "embeddings"
        else:
            key = "head"
        per_layer.setdefault(key, []).append(D.flatten())
    return {k: float(torch.cat(v).median()) for k, v in per_layer.items()}


def train(task, mode, lr, wd, seed, device):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import transformers
    transformers.logging.set_verbosity_error()

    cfg = TASKS[task]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ds = load_dataset("nyu-mll/glue", task)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    problem = "regression" if cfg["n_labels"] == 1 else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=cfg["n_labels"], problem_type=problem).to(device)
    opt = AdamThreeModes(model.parameters(), lr=lr, wd=wd, mode=mode)

    lt = make_loader(ds["train"], tok, cfg["keys"], cfg["bs"], True, seed)
    lv = make_loader(ds["validation"], tok, cfg["keys"], 64, False)
    total = cfg["epochs"] * len(lt)
    warmup = int(0.06 * total)

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        return lr * max(0.0, (total - step) / max(total - warmup, 1))

    hist = {"task": task, "mode": mode, "lr": lr, "wd": wd, "seed": seed,
            "val_metric": [], "train_loss": []}
    step = 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        run_loss, nb = 0.0, 0
        for enc, labels in lt:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device)
            if cfg["n_labels"] == 1:
                labels = labels.float()
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc, labels=labels)
            out.loss.backward()
            opt.step()
            run_loss += float(out.loss.detach())
            nb += 1
            step += 1
        # валидация
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for enc, labels in lv:
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits
                if cfg["n_labels"] == 1:
                    ps.extend(logits.squeeze(-1).float().cpu().tolist())
                else:
                    ps.extend(logits.argmax(-1).cpu().tolist())
                ys.extend(labels.tolist())
        m = metric_value(cfg["metric"], ys, ps)
        hist["val_metric"].append(m)
        hist["train_loss"].append(run_loss / max(nb, 1))
    hist["best_metric"] = max(hist["val_metric"])
    hist["final_metric"] = hist["val_metric"][-1]
    hist["dhat_by_layer"] = layer_dhat_stats(model, opt)
    hist["wall_s"] = time.time() - t0
    fname = f"{task}_{mode}_lr{lr:g}_wd{wd:g}_s{seed}.json"
    with open(os.path.join(RES, fname), "w") as f:
        json.dump(hist, f)
    print(f"saved {fname} best={hist['best_metric']:.4f} ({hist['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--mode", required=True, choices=["l2", "w", "wh"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    train(args.task, args.mode, args.lr, args.wd, args.seed, args.device)
