#!/usr/bin/env python
"""Tier-2 batch 1: (1) random-edge vs gene-disjoint PR-AUC for our model,
(2) gene-degree/identity baseline under both splits, (3) feature-category ablations.
Reuses sweep_union features + reference labels. -> results/analysis/tier2_batch1.txt"""
import json, numpy as np, pandas as pd, os
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
np.random.seed(0); OUT="results/analysis"; os.makedirs(OUT,exist_ok=True); F16MAX=65504.0

U=pd.read_parquet("data/processed/pairs/sweep_union.parquet")
GM=pd.read_parquet("data/processed/features/gene_matrix.parquet")
gcols=[c for c in GM.columns if c!="entrez"]
def fam(c):
    if c.startswith("e") and c[1:].isdigit(): return "seq"      # ESM
    if c.startswith("go"): return "curated"                     # GO
    if c.startswith("pfam"): return "domain"                    # Pfam
    if c in ("degree","log_degree","mean_string","mean_nsrc"): return "network"
    if c.startswith("mut") or c.startswith("cnv"): return "genomic"
    return "other"
gene_fam=[fam(c) for c in gcols]
featcols=[c for c in U.columns if c.split("_")[0] in ("min","max","absdiff","prod")]
gene_dim=[int(c.split("_")[1]) for c in featcols]
feat_fam=np.array([gene_fam[g] for g in gene_dim])
X=np.nan_to_num(U[featcols].to_numpy(np.float32),posinf=F16MAX,neginf=-F16MAX)

RT=U.r_tumor.values;RN=U.r_normal.values;drv=U.delta_r.values;sup=U.support_cancers.values
lost=(sup>=3)&(RN>=0.5)&(RT<0.3)&((RN-RT)>=0.3); stable=(sup>=3)&(np.abs(drv)<0.1)&(np.minimum(RT,RN)>=0.5)
y=np.full(len(U),-1); y[stable]=0; y[lost]=1; m=y>=0
idx=np.where(m)[0]; yl=y[idx]; Xl=X[idx]; ga=U.gene_a.values[idx]; gb=U.gene_b.values[idx]
gf=pd.read_parquet("data/processed/features/gene_features.parquet")[["entrez","degree"]]
deg=dict(zip(gf.entrez.astype(int),gf.degree))
print(f"labeled: {len(idx)} pos={int(yl.sum())}",flush=True)

def folds_gene(seed=0):
    genes=np.array(sorted(set(ga)|set(gb))); rng=np.random.default_rng(seed); rng.shuffle(genes)
    fo={int(g):i%5 for i,g in enumerate(genes)}
    fa=np.array([fo[int(g)] for g in ga]); fb=np.array([fo[int(g)] for g in gb])
    return [( (fa!=f)&(fb!=f), (fa==f)&(fb==f) ) for f in range(5)]   # (train,test) both-in-fold test
def folds_edge(seed=0):
    rng=np.random.default_rng(seed); ef=rng.integers(0,5,len(idx))
    return [(ef!=f, ef==f) for f in range(5)]

def evalsplit(getX, folds, model="xgb"):
    prs=[]
    for tr,te in folds:
        if te.sum()<10 or yl[te].sum()<2 or (yl[tr]==1).sum()<2: continue
        Xtr,Xte=getX(tr),getX(te)
        if model=="xgb":
            spw=(yl[tr]==0).sum()/max((yl[tr]==1).sum(),1)
            clf=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
                colsample_bytree=0.5,scale_pos_weight=spw,n_jobs=-1,tree_method="hist",random_state=0)
        else:
            clf=LogisticRegression(max_iter=1000,class_weight="balanced")
        clf.fit(Xtr,yl[tr]); p=clf.predict_proba(Xte)[:,1]; prs.append(average_precision_score(yl[te],p))
    return (np.mean(prs),np.std(prs)) if prs else (np.nan,np.nan)

base=yl.mean()
print(f"\n=== (1) RANDOM-EDGE vs GENE-DISJOINT (our full model, XGBoost) ===")
re_m,re_s=evalsplit(lambda s:Xl[s], folds_edge())
gd_m,gd_s=evalsplit(lambda s:Xl[s], folds_gene())
print(f"  full features   random-edge PR-AUC={re_m:.3f}+/-{re_s:.3f}   gene-disjoint={gd_m:.3f}+/-{gd_s:.3f}   (baseline {base:.3f})")

print(f"\n=== (2) gene-DEGREE baseline (2 features: endpoint degrees) under both splits ===")
Dg=np.stack([[deg.get(int(a),0),deg.get(int(b),0)] for a,b in zip(ga,gb)]).astype(float)
rd_m,_=evalsplit(lambda s:Dg[s], folds_edge()); gg_m,_=evalsplit(lambda s:Dg[s], folds_gene())
print(f"  degree-only     random-edge PR-AUC={rd_m:.3f}   gene-disjoint={gg_m:.3f}")
print(f"  (if degree-only is high under random-edge but collapses under gene-disjoint,")
print(f"   trivial gene priors explain much of the random-split performance)")

print(f"\n=== (3) feature-category ablation (gene-disjoint) ===")
cats={"seq(ESM)":["seq"],"domain(Pfam)":["domain"],"curated(GO)":["curated"],
      "network":["network"],"genomic":["genomic"],
      "seq+domain":["seq","domain"],"seq+genomic":["seq","genomic"],
      "non-curated(all\\GO)":["seq","domain","network","genomic"],
      "all":["seq","domain","curated","network","genomic"]}
print(f"  {'category':22s}{'nfeat':>7s}{'PR-AUC':>9s}{'lift':>6s}")
rows=[]
for name,fams in cats.items():
    cm=np.isin(feat_fam,fams)
    if cm.sum()==0: continue
    mmean,_=evalsplit(lambda s:Xl[np.ix_(s,cm)] if False else Xl[s][:,cm], folds_gene())
    print(f"  {name:22s}{int(cm.sum()):7d}{mmean:9.3f}{mmean/base:6.1f}",flush=True)
    rows.append((name,int(cm.sum()),round(mmean,3),round(mmean/base,2)))

print(f"\n=== logistic-regression baseline (full features, gene-disjoint) ===")
lr_m,_=evalsplit(lambda s:Xl[s], folds_gene(), model="lr")
print(f"  logistic regression PR-AUC={lr_m:.3f}")
pd.DataFrame(rows,columns=["category","nfeat","prauc","lift"]).to_csv(f"{OUT}/ablation_categories.tsv",sep="\t",index=False)
json.dump({"random_edge":re_m,"gene_disjoint":gd_m,"baseline":float(base),
           "degree_random":rd_m,"degree_disjoint":gg_m,"logreg":lr_m},
          open(f"{OUT}/tier2_batch1.json","w"),indent=2)
print(f"\nwrote {OUT}/tier2_batch1.json + ablation_categories.tsv")
