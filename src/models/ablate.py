#!/usr/bin/env python
"""Feature-family ablation for both models, protein-disjoint 5-fold CV, XGBoost.
Variants: all, no_GO, ESM_only, no_ESM, bio_only. -> results/model/ablation.tsv"""
import os, glob, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
np.random.seed(0)
PAIRS="data/processed/pairs/labeled"; OUT=os.environ.get("MODEL_OUT","results/model"); os.makedirs(OUT,exist_ok=True)
F16MAX=65504.0; NFOLD=5

GM=pd.read_parquet("data/processed/features/gene_matrix.parquet")
gcols=[c for c in GM.columns if c!="entrez"]
def fam_of(c):
    if c.startswith("e") and c[1:].isdigit(): return "ESM"
    if c.startswith("go"): return "GO"
    if c.startswith("pfam"): return "Pfam"
    if c in ("degree","log_degree","mean_string","mean_nsrc"): return "Network"
    if c.startswith("mut"): return "Mutation"
    if c.startswith("cnv"): return "CNV"
    return "Other"
gene_fam=np.array([fam_of(c) for c in gcols])

df=pd.concat([pd.read_parquet(p) for p in glob.glob(f"{PAIRS}/*.parquet")],ignore_index=True)
feat=[c for c in df.columns if c.split("_")[0] in ("min","max","absdiff","prod")]
X=np.nan_to_num(df[feat].to_numpy(np.float32),nan=0.0,posinf=F16MAX,neginf=-F16MAX)
feat_fam=np.array([gene_fam[int(c.split("_")[1])] for c in feat])

VARIANTS={
 "all":        lambda f: np.ones(len(f),bool),
 "no_GO":      lambda f: f!="GO",
 "ESM_only":   lambda f: f=="ESM",
 "no_ESM":     lambda f: f!="ESM",
 "bio_only":   lambda f: np.isin(f,["Mutation","CNV","Network"]),
}
genes=np.array(sorted(set(df.gene_a)|set(df.gene_b)))
rng=np.random.default_rng(0); rng.shuffle(genes); fo={g:i%NFOLD for i,g in enumerate(genes)}
fa=df.gene_a.map(fo).values; fb=df.gene_b.map(fo).values
gf=pd.read_parquet("data/processed/features/gene_features.parquet")[["entrez","degree"]]
dmap=dict(zip(gf.entrez.astype(int),gf.degree))
deg=np.array([dmap.get(int(a),0)+dmap.get(int(b),0) for a,b in zip(df.gene_a,df.gene_b)],float)

def dmatch(pos_idx,neg_idx):
    r=np.random.default_rng(0); pd_=np.log1p(deg[pos_idx]); nd=np.log1p(deg[neg_idx])
    b=np.quantile(pd_,[0,.2,.4,.6,.8,1.0]); keep=list(pos_idx)
    for lo,hi in zip(b[:-1],b[1:]):
        pool=neg_idx[(nd>=lo)&(nd<=hi)]; npos=((pd_>=lo)&(pd_<=hi)).sum(); t=min(len(pool),npos*8)
        if t>0: keep+=list(r.choice(pool,t,replace=False))
    return np.array(sorted(set(keep)))

def evaluate(idx,y,colmask):
    Xi=X[np.ix_(idx,colmask)]; yi=y[idx]; f_a=fa[idx]; f_b=fb[idx]; prs=[]; rocs=[]
    for f in range(NFOLD):
        te=(f_a==f)&(f_b==f); tr=(f_a!=f)&(f_b!=f)
        if te.sum()<10 or yi[te].sum()<2: continue
        spw=(yi[tr]==0).sum()/max((yi[tr]==1).sum(),1)
        clf=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
            colsample_bytree=0.5,scale_pos_weight=spw,n_jobs=-1,tree_method="hist",random_state=0)
        clf.fit(Xi[tr],yi[tr]); p=clf.predict_proba(Xi[te])[:,1]
        prs.append(average_precision_score(yi[te],p)); rocs.append(roc_auc_score(yi[te],p))
    return np.mean(prs),np.std(prs),np.mean(rocs)

def main():
    rows=[]
    yA=np.where(df.class_lost=="lost",1,np.where(df.class_lost=="stable",0,-1))
    idxA=np.where(yA>=0)[0]; yA01=(yA==1).astype(int)
    yB=np.where(df.class_gained=="gained",1,np.where(df.class_gained=="stable",0,-1))
    posB=np.where((yB==1))[0]; negB=np.where((yB==0))[0]; idxB=dmatch(posB,negB); yB01=(yB==1).astype(int)
    for vname,vf in VARIANTS.items():
        cm=vf(feat_fam)
        if cm.sum()==0: continue
        pA,sA,rA=evaluate(idxA,yA01,cm)
        pB,sB,rB=evaluate(idxB,yB01,cm)
        rows.append(("A_lost",vname,int(cm.sum()),pA,sA,rA))
        rows.append(("B_gained",vname,int(cm.sum()),pB,sB,rB))
        print(f"{vname:10s} nfeat={cm.sum():5d} | A PR={pA:.3f}±{sA:.3f} ROC={rA:.3f} | B PR={pB:.3f}±{sB:.3f} ROC={rB:.3f}",flush=True)
    R=pd.DataFrame(rows,columns=["model","variant","n_feat","PR_AUC","PR_std","ROC_AUC"])
    R.to_csv(f"{OUT}/ablation.tsv",sep="\t",index=False)
    print(f"\nwrote {OUT}/ablation.tsv",flush=True)

if __name__=="__main__": main()
