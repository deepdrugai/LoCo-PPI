#!/usr/bin/env python
"""Stream per-edge symmetric pair features to partitioned parquet.
Per gene-dim: [min(A,B), max(A,B), |A-B|, A*B]  (permutation-invariant).
Args: <edge_source: labeled|all> <out_subdir>"""
import sys, os, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

F="data/processed/features"
GM=pd.read_parquet(f"{F}/gene_matrix.parquet")
gidx={int(e):i for i,e in enumerate(GM.entrez)}
X=GM.drop(columns=["entrez"]).to_numpy(dtype=np.float32)   # genes x D
D=X.shape[1]; print(f"gene matrix: {X.shape[0]} x {D}",flush=True)

src=sys.argv[1]; outdir=f"data/processed/pairs/{sys.argv[2]}"; os.makedirs(outdir,exist_ok=True)
if src=="labeled":
    lab=pd.read_parquet("data/processed/labels/labels_master.parquet")
    lab=lab[(lab.class_lost!="na")|(lab.class_gained!="na")]
    edges=lab[["gene_a","gene_b"]].to_numpy(dtype=np.int64)
    meta=lab[["gene_a","gene_b","class_lost","class_gained","label","necare_label"]].reset_index(drop=True)
else:
    e=pd.read_parquet("data/interim/ppi_edges.parquet",columns=["gene_a","gene_b"])
    edges=e.to_numpy(dtype=np.int64); meta=e.reset_index(drop=True)

names=[f"{op}_{i}" for op in ("min","max","absdiff","prod") for i in range(D)]
CHUNK=50000; N=len(edges); print(f"edges: {N}  out-dims: {len(names)}",flush=True)
part=0
for s in range(0,N,CHUNK):
    ea=edges[s:s+CHUNK,0]; eb=edges[s:s+CHUNK,1]
    ia=np.array([gidx.get(int(g),-1) for g in ea]); ib=np.array([gidx.get(int(g),-1) for g in eb])
    A=np.where(ia[:,None]>=0, X[np.clip(ia,0,None)], 0.0)
    B=np.where(ib[:,None]>=0, X[np.clip(ib,0,None)], 0.0)
    blk=np.concatenate([np.minimum(A,B),np.maximum(A,B),np.abs(A-B),A*B],axis=1).astype(np.float16)
    df=pd.DataFrame(blk,columns=names)
    m=meta.iloc[s:s+CHUNK].reset_index(drop=True)
    for c in m.columns: df[c]=m[c].values
    pq.write_table(pa.Table.from_pandas(df),f"{outdir}/part-{part:05d}.parquet",
                   compression="zstd",use_dictionary=False)
    part+=1
    if part%5==0 or s+CHUNK>=N: print(f"  {min(s+CHUNK,N)}/{N}  ({part} parts)",flush=True)
print(f"done -> {outdir}/  ({part} parts)",flush=True)
