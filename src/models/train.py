#!/usr/bin/env python
"""Two binary models (A=lost-vs-stable, B=gained-vs-stable), RF + XGBoost,
5-fold protein-disjoint CV. Model B uses degree-matched stable negatives.
Reports PR-AUC / ROC-AUC per fold; saves metrics, predictions, feature importance."""
import os, sys, glob, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
np.random.seed(0)
PAIRS="data/processed/pairs/labeled"; OUT=os.environ.get("MODEL_OUT","results/model"); os.makedirs(OUT,exist_ok=True)
F16MAX=65504.0; NFOLD=5

def load():
    df=pd.concat([pd.read_parquet(p) for p in glob.glob(f"{PAIRS}/*.parquet")],ignore_index=True)
    feat=[c for c in df.columns if c.split("_")[0] in ("min","max","absdiff","prod")]
    X=df[feat].to_numpy(dtype=np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=F16MAX,neginf=-F16MAX)   # clip inf from float16 prod
    return df,feat,X

def protein_folds(genes_a,genes_b,k):
    genes=np.array(sorted(set(genes_a)|set(genes_b)))
    rng=np.random.default_rng(0); rng.shuffle(genes)
    fold_of={g:i%k for i,g in enumerate(genes)}
    fa=np.array([fold_of[g] for g in genes_a]); fb=np.array([fold_of[g] for g in genes_b])
    return fa,fb  # edge is in test fold f iff both endpoints in f

def degree_match_neg(sub, pos_mask, deg, ratio=8, seed=0):
    rng=np.random.default_rng(seed)
    pos_deg=deg[pos_mask]
    neg_idx=np.where(~pos_mask)[0]
    # bin by log-degree, sample negatives matching positive degree distribution
    bins=np.quantile(np.log1p(pos_deg),[0,.2,.4,.6,.8,1.0])
    keep=[]
    negdeg=np.log1p(deg[neg_idx])
    for lo,hi in zip(bins[:-1],bins[1:]):
        pool=neg_idx[(negdeg>=lo)&(negdeg<=hi)]
        npos=((np.log1p(pos_deg)>=lo)&(np.log1p(pos_deg)<=hi)).sum()
        take=min(len(pool),npos*ratio)
        if take>0: keep+=list(rng.choice(pool,take,replace=False))
    return np.array(sorted(set(keep)))

def run_model(name, mask, y, X, fa, fb, deg, degree_match=False):
    idx=np.where(mask)[0]
    if degree_match:
        pos=y[idx]==1
        sel=degree_match_neg(idx, pos, deg[idx])
        # sel indexes into idx-space; combine positives + matched negs
        posi=np.where(pos)[0]; keep=np.array(sorted(set(list(posi)+list(sel))))
        idx=idx[keep]
    Xs=X[idx]; ys=y[idx]; f_a=fa[idx]; f_b=fb[idx]
    print(f"  {name}: {len(idx)} edges, pos={int(ys.sum())} neg={int((1-ys).sum())}",flush=True)
    res={"rf":[], "xgb":[]}
    oof={"rf":np.full(len(idx),np.nan), "xgb":np.full(len(idx),np.nan)}
    for f in range(NFOLD):
        te=(f_a==f)&(f_b==f)                    # both endpoints held out
        tr=(f_a!=f)&(f_b!=f)                    # neither endpoint in test fold
        if te.sum()<10 or ys[te].sum()<2: 
            print(f"    fold{f}: too few test pos, skip",flush=True); continue
        spw=(ys[tr]==0).sum()/max((ys[tr]==1).sum(),1)
        models={
          "rf": RandomForestClassifier(n_estimators=400,max_depth=None,n_jobs=-1,
                    class_weight="balanced",random_state=0),
          "xgb": XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.05,
                    subsample=0.8,colsample_bytree=0.5,scale_pos_weight=spw,
                    n_jobs=-1,eval_metric="aucpr",random_state=0,tree_method="hist"),
        }
        for m,clf in models.items():
            clf.fit(Xs[tr],ys[tr])
            p=clf.predict_proba(Xs[te])[:,1]; oof[m][te]=p
            pr=average_precision_score(ys[te],p); roc=roc_auc_score(ys[te],p)
            res[m].append((pr,roc,int(te.sum()),int(ys[te].sum())))
            print(f"    fold{f} {m}: PR-AUC={pr:.3f} ROC-AUC={roc:.3f} (n={te.sum()},pos={int(ys[te].sum())})",flush=True)
    summary={}
    for m in ("rf","xgb"):
        if res[m]:
            prs=[r[0] for r in res[m]]; rocs=[r[1] for r in res[m]]
            base=ys.mean()
            summary[m]={"PR_AUC_mean":float(np.mean(prs)),"PR_AUC_std":float(np.std(prs)),
                        "ROC_AUC_mean":float(np.mean(rocs)),"baseline_PR":float(base),
                        "lift":float(np.mean(prs)/base)}
            print(f"  == {name} {m}: PR-AUC {np.mean(prs):.3f}±{np.std(prs):.3f} "
                  f"(baseline {base:.3f}, lift {np.mean(prs)/base:.1f}x) ROC {np.mean(rocs):.3f}",flush=True)
    return summary

def main():
    df,feat,X=load()
    print(f"loaded {X.shape[0]} edges x {X.shape[1]} feats",flush=True)
    fa,fb=protein_folds(df.gene_a.values,df.gene_b.values,NFOLD)
    # per-edge degree = sum of endpoint degrees (from network features)
    gf=pd.read_parquet("data/processed/features/gene_features.parquet")[["entrez","degree"]]
    dmap=dict(zip(gf.entrez.astype(int),gf.degree))
    deg=np.array([dmap.get(int(a),0)+dmap.get(int(b),0) for a,b in zip(df.gene_a,df.gene_b)],dtype=float)

    out={}
    # Model A: lost(1) vs stable(0)
    yA=np.where(df.class_lost=="lost",1,np.where(df.class_lost=="stable",0,-1))
    out["A_lost"]=run_model("A(lost-vs-stable)", yA>=0, (yA==1).astype(int), X, fa, fb, deg)
    # Model B: gained(1) vs stable(0), degree-matched
    yB=np.where(df.class_gained=="gained",1,np.where(df.class_gained=="stable",0,-1))
    out["B_gained"]=run_model("B(gained-vs-stable)", yB>=0, (yB==1).astype(int), X, fa, fb, deg, degree_match=True)

    json.dump(out,open(f"{OUT}/metrics.json","w"),indent=2)
    print(f"\nwrote {OUT}/metrics.json",flush=True)

if __name__=="__main__": main()
