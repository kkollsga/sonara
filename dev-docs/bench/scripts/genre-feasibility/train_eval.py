#!/usr/bin/env python3
"""P6 genre feasibility — step 5: train the multiclass MLP and evaluate.

Recipe = vocal_model.train generalized to N classes: pure numpy,
48 -> hidden ReLU -> N softmax, standardization folded into layer 1,
class-weighted cross-entropy, L2, fixed seed.

Protocol (preregistered in 2026-08-17-genre-feasibility-prereg.md):
  - variant selection (hidden in {16,32,64}) by 5-fold artist-grouped CV
    macro-F1 on the TRAIN split only, on the mix (cached+fresh) train set;
  - final models: (a) cached-only train, (b) cached+fresh mix train;
  - eval once per final model on eval-cached and eval-fresh;
  - the bar is judged on the MIX-trained model's FRESH-eval numbers.

Outputs: dev-docs/bench/out/genre-feasibility/metrics.json,
         model_mix.json (vocal_model-style JSON, genre labels),
         stdout log with tables.
"""

import json
from collections import defaultdict

import numpy as np

OUTDIR = ("/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/out/"
          "genre-feasibility")
SEED = 0
EPOCHS, LR, L2 = 3500, 0.05, 1e-3
HIDDEN_VARIANTS = (16, 32, 64)


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_mlp(X, yi, k, *, hidden, epochs=EPOCHS, lr=LR, l2=L2, seed=SEED):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    Y = np.zeros((n, k))
    Y[np.arange(n), yi] = 1.0
    counts = np.bincount(yi, minlength=k).astype(np.float64)
    w_sample = (n / (k * counts))[yi][:, None]
    rng = np.random.default_rng(seed)
    h = int(hidden)
    W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, k))
    b2 = np.zeros(k)
    for _ in range(int(epochs)):
        Z1 = Xs @ W1 + b1
        A1 = np.maximum(Z1, 0.0)
        P = softmax_rows(A1 @ W2 + b2)
        dZ2 = (P - Y) * w_sample / n
        dW2 = A1.T @ dZ2 + l2 * W2
        db2 = dZ2.sum(axis=0)
        dZ1 = (dZ2 @ W2.T) * (Z1 > 0.0)
        dW1 = Xs.T @ dZ1 + l2 * W1
        db1 = dZ1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    W1_raw = W1 / sigma[:, None]
    b1_raw = b1 - (mu / sigma) @ W1
    return [{"W": W1_raw, "b": b1_raw, "activation": "relu"},
            {"W": W2, "b": b2, "activation": "softmax"}]


def predict(layers, X):
    A = np.asarray(X, dtype=np.float64)
    for layer in layers:
        Z = A @ layer["W"] + layer["b"]
        A = np.maximum(Z, 0.0) if layer["activation"] == "relu" else Z
    return softmax_rows(A).argmax(axis=1)


def per_class_f1(y_true, y_pred, k):
    f1s = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return f1s


def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def artist_folds(artists, n_folds=5, seed=42):
    uniq = sorted(set(artists))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    groups = np.array_split(np.array(uniq), n_folds)
    a = np.asarray(artists)
    return [np.isin(a, g) for g in groups]


def main():
    pool = [json.loads(l) for l in open(f"{OUTDIR}/pool.jsonl",
                                        encoding="utf-8")]
    fresh = [json.loads(l) for l in open(f"{OUTDIR}/fresh.jsonl",
                                         encoding="utf-8")]
    labels = sorted({r["label"] for r in pool})
    k = len(labels)
    li = {c: i for i, c in enumerate(labels)}
    print(f"classes ({k}): {labels}")

    def mat(rows):
        X = np.array([r["emb"] for r in rows])
        y = np.array([li[r["label"]] for r in rows])
        a = [r["artist"] for r in rows]
        return X, y, a

    tr_cached = [r for r in pool if r["split"] == "train"]
    ev_cached = [r for r in pool if r["split"] == "eval"]
    tr_fresh = [r for r in fresh if r["split"] == "train"]
    ev_fresh = [r for r in fresh if r["split"] == "eval"]
    print(f"train cached {len(tr_cached)}, fresh {len(tr_fresh)}; "
          f"eval cached {len(ev_cached)}, fresh {len(ev_fresh)}")

    Xtc, ytc, atc = mat(tr_cached)
    Xec, yec, _ = mat(ev_cached)
    Xtf, ytf, atf = mat(tr_fresh)
    Xef, yef, _ = mat(ev_fresh)
    Xmix = np.vstack([Xtc, Xtf])
    ymix = np.concatenate([ytc, ytf])
    amix = atc + atf

    # ---- variant selection: artist-grouped 5-fold CV on mix train ----
    print("\n== 5-fold artist-grouped CV on train (mix) ==")
    best = None
    cv_results = {}
    for hidden in HIDDEN_VARIANTS:
        scores = []
        for mask in artist_folds(amix):
            layers = train_mlp(Xmix[~mask], ymix[~mask], k, hidden=hidden)
            pred = predict(layers, Xmix[mask])
            scores.append(float(np.mean(per_class_f1(ymix[mask], pred, k))))
        m = float(np.mean(scores))
        cv_results[hidden] = {"mean_macro_f1": m,
                              "folds": [round(s, 4) for s in scores]}
        print(f"  hidden={hidden:3d}  CV macro-F1 {m:.4f}  "
              f"(folds {', '.join(f'{s:.3f}' for s in scores)})")
        if best is None or m > best[1]:
            best = (hidden, m)
    hidden = best[0]
    print(f"selected hidden={hidden}")

    # ---- final models ----
    model_cached = train_mlp(Xtc, ytc, k, hidden=hidden)
    model_mix = train_mlp(Xmix, ymix, k, hidden=hidden)

    results = {"labels": labels, "hidden": hidden, "cv": cv_results,
               "eval": {}}

    def report(name, layers, X, y):
        pred = predict(layers, X)
        f1s = per_class_f1(y, pred, k)
        macro = float(np.mean(f1s))
        cm = confusion(y, pred, k)
        acc = float((pred == y).mean())
        print(f"\n== {name} ==  macro-F1 {macro:.4f}  acc {acc:.4f}")
        for c, f in zip(labels, f1s):
            n = int((y == li[c]).sum())
            print(f"  {c:18s} F1 {f:.3f}  (n={n})")
        results["eval"][name] = {
            "macro_f1": macro, "accuracy": acc,
            "per_class_f1": {c: round(f, 4) for c, f in zip(labels, f1s)},
            "confusion_rows_true": cm.tolist(),
        }
        return macro

    report("cached-trained / eval-cached", model_cached, Xec, yec)
    report("cached-trained / eval-FRESH", model_cached, Xef, yef)
    report("mix-trained / eval-cached", model_mix, Xec, yec)
    report("mix-trained / eval-FRESH (DECIDES)", model_mix, Xef, yef)

    with open(f"{OUTDIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # vocal_model-style JSON for P7 reference
    model_json = {
        "id": "genre-feasibility-p6-mix",
        "embedding_version": 2,
        "labels": labels,
        "layers": [{"weights": l["W"].T.tolist(),
                    "bias": np.asarray(l["b"]).ravel().tolist(),
                    "activation": l["activation"]} for l in model_mix],
    }
    with open(f"{OUTDIR}/model_mix.json", "w", encoding="utf-8") as f:
        json.dump(model_json, f)
    print(f"\nwrote {OUTDIR}/metrics.json and model_mix.json")


if __name__ == "__main__":
    main()
