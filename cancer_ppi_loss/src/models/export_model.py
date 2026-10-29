#!/usr/bin/env python
"""Fit the final models on all labeled pairs and save them.
Model A: lost vs stable. Model B: gained vs stable, with degree-matched stable negatives.
Same features and hyperparameters as the cross-validation in train.py."""
import os, glob, numpy as np, pandas as pd
from xgboost import XGBClassifier
np.random.seed(0)
PAIRS = "data/processed/pairs/labeled"
OUT = os.environ.get("MODEL_OUT", "results/model"); os.makedirs(OUT, exist_ok=True)
F16MAX = 65504.0

def load():
    df = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{PAIRS}/*.parquet")], ignore_index=True)
    feat = [c for c in df.columns if c.split("_")[0] in ("min", "max", "absdiff", "prod")]
    X = df[feat].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=F16MAX, neginf=-F16MAX)
    return df, feat, X

def degree_match_neg(pos_mask, deg, ratio=8, seed=0):
    rng = np.random.default_rng(seed)
    pos_deg = deg[pos_mask]; neg_idx = np.where(~pos_mask)[0]
    bins = np.quantile(np.log1p(pos_deg), [0, .2, .4, .6, .8, 1.0]); keep = []
    negdeg = np.log1p(deg[neg_idx])
    for lo, hi in zip(bins[:-1], bins[1:]):
        pool = neg_idx[(negdeg >= lo) & (negdeg <= hi)]
        npos = ((np.log1p(pos_deg) >= lo) & (np.log1p(pos_deg) <= hi)).sum()
        take = min(len(pool), npos * ratio)
        if take > 0: keep += list(rng.choice(pool, take, replace=False))
    return np.array(sorted(set(keep)))

def fit_save(name, X, y, feat):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.5, scale_pos_weight=spw, n_jobs=-1,
                        eval_metric="aucpr", random_state=0, tree_method="hist")
    clf.fit(X, y)
    clf.save_model(f"{OUT}/model_{name}.ubj")
    pd.Series(feat).to_csv(f"{OUT}/model_{name}.features.txt", index=False, header=False)
    print(f"saved {OUT}/model_{name}.ubj  ({len(feat)} features, {int(y.sum())} pos / {int((1 - y).sum())} neg)")

def main():
    df, feat, X = load()
    print(f"loaded {X.shape[0]} pairs x {X.shape[1]} features", flush=True)
    gf = pd.read_parquet("data/processed/features/gene_features.parquet")[["entrez", "degree"]]
    dmap = dict(zip(gf.entrez.astype(int), gf.degree))
    deg = np.array([dmap.get(int(a), 0) + dmap.get(int(b), 0) for a, b in zip(df.gene_a, df.gene_b)], dtype=float)
    yA = np.where(df.class_lost == "lost", 1, np.where(df.class_lost == "stable", 0, -1))
    mA = yA >= 0
    fit_save("A_lost", X[mA], (yA[mA] == 1).astype(int), feat)
    yB = np.where(df.class_gained == "gained", 1, np.where(df.class_gained == "stable", 0, -1))
    idxB = np.where(yB >= 0)[0]; posB = (yB[idxB] == 1)
    sel = degree_match_neg(posB, deg[idxB])
    keep = np.array(sorted(set(list(np.where(posB)[0]) + list(sel))))
    idxB = idxB[keep]
    fit_save("B_gained", X[idxB], (yB[idxB] == 1).astype(int), feat)

if __name__ == "__main__":
    main()
