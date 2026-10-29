#!/usr/bin/env python
"""McNemar's paired test: our XGBoost vs NECARE on IDENTICAL protein-disjoint test edges,
NECARE gold task. The paper's core statistic."""
import pandas as pd, numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
X=pd.read_csv("results/necare/xgb_predictions.tsv",sep="\t")
Nc=pd.read_csv("results/necare/necare_predictions.tsv",sep="\t")
# align on the edge (ea,eb) + true label
m=X.merge(Nc,on=["ea","eb"],suffixes=("","_n"))
m=m.dropna(subset=["xgb_pred","necare_pred"])
y=m.y.values.astype(int)
xp=m.xgb_pred.values.astype(int); npd=m.necare_pred.values.astype(int)
print(f"paired edges (both methods predicted, identical folds): {len(m)}  pos={y.sum()}")

xok=(xp==y); nok=(npd==y)
# 2x2 McNemar table
both=int((xok&nok).sum()); xonly=int((xok&~nok).sum()); nonly=int((~xok&nok).sum()); neither=int((~xok&~nok).sum())
print("\n=== paired outcome table ===")
print(f"  both correct           : {both}")
print(f"  XGB-only correct  (b)  : {xonly}   <- edges our method gets that NECARE misses")
print(f"  NECARE-only correct(c) : {nonly}   <- edges NECARE gets that we miss")
print(f"  both wrong             : {neither}")
tbl=[[both,xonly],[nonly,neither]]
res=mcnemar(tbl, exact=(min(xonly,nonly)<25))
print(f"\nMcNemar's: statistic={res.statistic:.3f}  p={res.pvalue:.3e}")
print("  -> significant (p<0.05):", res.pvalue<0.05, "| direction:", "XGB>NECARE" if xonly>nonly else "NECARE>XGB")

print("\n=== overall metrics (pooled OOF, identical edges) ===")
print(f"  XGBoost : MCC={matthews_corrcoef(y,xp):.3f} F1={f1_score(y,xp):.3f} AUC={roc_auc_score(y,m.xgb_score):.3f}")
print(f"  NECARE  : MCC={matthews_corrcoef(y,npd):.3f} F1={f1_score(y,npd):.3f} AUC={roc_auc_score(y,m.necare_score):.3f}")
m.to_csv("results/necare/paired_predictions.tsv",sep="\t",index=False)
print("\nwrote results/necare/paired_predictions.tsv")
