#!/usr/bin/env python
"""Head-to-head part 1: our XGBoost on NECARE's gold task, 5-fold protein-disjoint.
Writes the SHARED fold assignment (both methods use it) + per-edge XGB predictions."""
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
np.random.seed(0); F16MAX=65504.0
OUT="results/necare"; import os; os.makedirs(OUT,exist_ok=True)

gold=pd.read_csv("results/necare_gold_common.tsv",sep="\t")  # ea, eb, y
gmx=pd.read_parquet("data/processed/features/gene_matrix.parquet").set_index("entrez")
gidx={int(e):i for i,e in enumerate(gmx.index)}
X=np.nan_to_num(gmx.to_numpy(np.float32),posinf=F16MAX,neginf=-F16MAX)
D=X.shape[1]

# symmetric pair features (same operators as main pipeline)
def pairfeat(ea,eb):
    ia=np.array([gidx[g] for g in ea]); ib=np.array([gidx[g] for g in eb])
    A=X[ia]; B=X[ib]
    return np.concatenate([np.minimum(A,B),np.maximum(A,B),np.abs(A-B),A*B],axis=1)

# SHARED protein-disjoint folds: partition the 1099 genes into 5
genes=np.array(sorted(set(gold.ea)|set(gold.eb)))
rng=np.random.default_rng(0); rng.shuffle(genes)
fold_of={int(g):i%5 for i,g in enumerate(genes)}
pd.DataFrame([(g,f) for g,f in fold_of.items()],columns=["entrez","fold"]).to_csv(f"{OUT}/gene_folds.tsv",sep="\t",index=False)
gold["fa"]=gold.ea.map(fold_of); gold["fb"]=gold.eb.map(fold_of)
# edge in test fold f iff BOTH endpoints in f (protein-disjoint)
gold.to_csv(f"{OUT}/gold_with_folds.tsv",sep="\t",index=False)

Xall=pairfeat(gold.ea.values,gold.eb.values); y=gold.y.values
oof=np.full(len(gold),np.nan)
per_fold=[]
for f in range(5):
    te=(gold.fa==f)&(gold.fb==f); tr=(gold.fa!=f)&(gold.fb!=f)
    if te.sum()<5 or y[te].sum()<1: 
        print(f"fold {f}: too few test ({te.sum()}), skip"); continue
    spw=(y[tr]==0).sum()/max((y[tr]==1).sum(),1)
    clf=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.5,scale_pos_weight=spw,n_jobs=-1,tree_method="hist",random_state=0)
    clf.fit(Xall[tr],y[tr]); p=clf.predict_proba(Xall[te])[:,1]; oof[te.values]=p
    pred=(p>=0.5).astype(int)
    per_fold.append((f,int(te.sum()),int(y[te].sum()),
                     matthews_corrcoef(y[te],pred),roc_auc_score(y[te],p) if y[te].sum()<te.sum() else np.nan))
    print(f"fold {f}: n={te.sum()} pos={y[te].sum()} MCC={matthews_corrcoef(y[te],pred):.3f}")
gold["xgb_score"]=oof
import numpy as _np
gold["xgb_pred"]=_np.where(_np.isnan(oof), _np.nan, (oof>=0.5).astype(float))
gold.to_csv(f"{OUT}/xgb_predictions.tsv",sep="\t",index=False)
ok=~np.isnan(oof)
print(f"\n=== OUR XGBoost, protein-disjoint, NECARE gold task ===")
print(f"  overall MCC={matthews_corrcoef(y[ok],(oof[ok]>=0.5).astype(int)):.3f} "
      f"AUC={roc_auc_score(y[ok],oof[ok]):.3f} F1={f1_score(y[ok],(oof[ok]>=0.5).astype(int)):.3f} (n={ok.sum()})")
print(f"wrote {OUT}/xgb_predictions.tsv + gene_folds.tsv")
