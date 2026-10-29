#!/usr/bin/env python
"""Inductive GraphSAGE for Model A (lost vs stable), 5-fold protein-disjoint.
Node features = gene_matrix (3277-d). Test-gene edges removed from message-passing
graph (inductive, no leakage). Edge classifier on symmetric endpoint embeddings.
Comparable to the XGBoost 0.62 baseline. -> results/model/gnn_metrics.json"""
import os, glob, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score, roc_auc_score
torch.manual_seed(0); np.random.seed(0)
dev="cuda" if torch.cuda.is_available() else "cpu"
OUT="results/model"; os.makedirs(OUT,exist_ok=True); F16MAX=65504.0; NFOLD=5; EPOCHS=60

# ---- node features ----
gm=pd.read_parquet("data/processed/features/gene_matrix.parquet")
gidx={int(e):i for i,e in enumerate(gm.entrez)}
Xn=np.nan_to_num(gm.drop(columns=["entrez"]).to_numpy(np.float32),posinf=F16MAX,neginf=-F16MAX)
Xn=torch.tensor(Xn); N,Fdim=Xn.shape
print(f"nodes {N} x {Fdim}", flush=True)

# ---- backbone graph (undirected edge_index) ----
bb=pd.read_parquet("data/interim/ppi_edges.parquet",columns=["gene_a","gene_b"])
ea=bb.gene_a.map(gidx).values; eb=bb.gene_b.map(gidx).values
E=np.stack([ea,eb]); E=np.concatenate([E,E[::-1]],axis=1)  # symmetrize
edge_all=torch.tensor(E,dtype=torch.long)

# ---- labeled edges (Model A) ----
lab=pd.concat([pd.read_parquet(p) for p in glob.glob("data/processed/pairs/labeled/*.parquet")],
              ignore_index=True) if False else pd.read_parquet("data/processed/labels/labels_master.parquet")
A=lab[lab.class_lost.isin(["lost","stable"])].copy()
A["ia"]=A.gene_a.map(gidx); A["ib"]=A.gene_b.map(gidx)
A=A.dropna(subset=["ia","ib"]); A["ia"]=A.ia.astype(int); A["ib"]=A.ib.astype(int)
A["y"]=(A.class_lost=="lost").astype(int)
print(f"Model A edges {len(A)} pos {int(A.y.sum())}", flush=True)

# ---- protein-disjoint folds ----
genes=np.array(sorted(set(A.gene_a)|set(A.gene_b)))
rng=np.random.default_rng(0); rng.shuffle(genes); fo={g:i%NFOLD for i,g in enumerate(genes)}
A["fa"]=A.gene_a.map(fo); A["fb"]=A.gene_b.map(fo)
# node-level fold for inductive graph masking
node_fold=np.full(N,-1)
for g,f in fo.items():
    if g in gidx: node_fold[gidx[g]]=f

class SAGE(nn.Module):
    def __init__(s,fin,h=256,emb=128):
        super().__init__(); s.c1=SAGEConv(fin,h); s.c2=SAGEConv(h,emb)
        s.mlp=nn.Sequential(nn.Linear(emb*4,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))
    def encode(s,x,ei):
        h=Fn.relu(s.c1(x,ei)); h=Fn.dropout(h,0.3,s.training); return s.c2(h,ei)
    def score(s,z,ia,ib):
        za,zb=z[ia],z[ib]
        feat=torch.cat([torch.min(za,zb),torch.max(za,zb),(za-zb).abs(),za*zb],dim=1)
        return s.mlp(feat).squeeze(-1)

def run_fold(f):
    te=((A.fa==f)&(A.fb==f)).values; tr=((A.fa!=f)&(A.fb!=f)).values
    if te.sum()<10 or A.y.values[te].sum()<2: return None
    # inductive graph: drop edges touching any test-fold node
    keep_node=(node_fold!=f)
    em=edge_all.numpy()
    mask=keep_node[em[0]]&keep_node[em[1]]
    ei_train=torch.tensor(em[:,mask],dtype=torch.long).to(dev)
    ei_full=edge_all.to(dev)   # for inference, test nodes see only train neighbors -> use masked too
    x=Xn.to(dev)
    ia=torch.tensor(A.ia.values).to(dev); ib=torch.tensor(A.ib.values).to(dev)
    y=torch.tensor(A.y.values,dtype=torch.float32).to(dev)
    tr_t=torch.tensor(np.where(tr)[0]).to(dev); te_t=torch.tensor(np.where(te)[0]).to(dev)
    pos_w=torch.tensor([(y[tr_t]==0).sum()/max((y[tr_t]==1).sum(),1)]).to(dev)

    m=SAGE(Fdim).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        z=m.encode(x,ei_train)                       # aggregate on train-only graph
        logit=m.score(z,ia[tr_t],ib[tr_t])
        loss=Fn.binary_cross_entropy_with_logits(logit,y[tr_t],pos_weight=pos_w)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        z=m.encode(x,ei_train)                       # test nodes embed from train neighbors (inductive)
        p=torch.sigmoid(m.score(z,ia[te_t],ib[te_t])).cpu().numpy()
    yt=A.y.values[te]
    pr=average_precision_score(yt,p); roc=roc_auc_score(yt,p)
    print(f"  fold{f}: PR-AUC={pr:.3f} ROC-AUC={roc:.3f} (n={te.sum()},pos={int(yt.sum())})",flush=True)
    return pr,roc

def main():
    print("device",dev,flush=True); prs=[];rocs=[]
    for f in range(NFOLD):
        r=run_fold(f)
        if r: prs.append(r[0]); rocs.append(r[1])
    base=A.y.mean()
    out={"model":"A_lost_GNN","PR_AUC_mean":float(np.mean(prs)),"PR_AUC_std":float(np.std(prs)),
         "ROC_AUC_mean":float(np.mean(rocs)),"baseline_PR":float(base),"lift":float(np.mean(prs)/base),
         "xgb_reference":0.618}
    json.dump(out,open(f"{OUT}/gnn_metrics.json","w"),indent=2)
    print(f"\n== GNN A(lost): PR-AUC {np.mean(prs):.3f}±{np.std(prs):.3f} "
          f"(baseline {base:.3f}, lift {np.mean(prs)/base:.1f}x) ROC {np.mean(rocs):.3f} "
          f"| XGB was 0.618 ==",flush=True)

if __name__=="__main__": main()
