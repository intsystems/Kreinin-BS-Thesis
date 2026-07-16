"""Подготовка данных для C2: FineWeb-Edu (sample-10BT) -> train.bin/val.bin (uint16, GPT-2 BPE).

Стримим датасет, токенизируем tiktoken'ом, пишем в memmap. Целевой объём задаётся --tokens.
Запуск: python -m theory_bench.fineweb_prep --tokens 3.2e9
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "fineweb")
os.makedirs(OUT, exist_ok=True)


def main(target_tokens, val_tokens=20_000_000):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    train_path = os.path.join(OUT, "train.bin")
    val_path = os.path.join(OUT, "val.bin")
    # пишем несжатыми блоками в обычные файлы (append), потом читаем memmap'ом
    ftr = open(train_path, "wb")
    fva = open(val_path, "wb")
    n_train, n_val = 0, 0
    buf = []
    t0 = time.time()
    BATCH = 1024
    it = iter(ds)
    while n_train < target_tokens:
        texts = []
        try:
            for _ in range(BATCH):
                texts.append(next(it)["text"])
        except StopIteration:
            if not texts:
                break
        ids_batch = enc.encode_ordinary_batch(texts, num_threads=8)
        flat = []
        for ids in ids_batch:
            flat.extend(ids)
            flat.append(eot)
        arr = np.array(flat, dtype=np.uint16)
        if n_val < val_tokens:
            fva.write(arr.tobytes())
            n_val += len(arr)
        else:
            ftr.write(arr.tobytes())
            n_train += len(arr)
        if (n_train // 100_000_000) != ((n_train - len(arr)) // 100_000_000):
            rate = (n_train + n_val) / max(time.time() - t0, 1e-9) / 1e6
            print(f"train={n_train/1e9:.2f}B val={n_val/1e6:.0f}M  {rate:.1f}M tok/s", flush=True)
    ftr.close()
    fva.close()
    print(f"DONE: train={n_train/1e9:.3f}B tokens, val={n_val/1e6:.1f}M tokens, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=3.2e9)
    args = ap.parse_args()
    main(int(args.tokens))
