#!/usr/bin/env python
"""Build symmetric pair features ONCE for the union of edges any threshold in the
sweep could label lost-or-stable. Reuses gene_matrix; carries stored r_tumor/r_normal/
delta_r/support so the sweep re-labels without re-correlating.
-> data/processed/pairs/sweep_union.parquet + sweep_union_meta.json"""
import json, numpy as np, pandas as pd, os
F16MAX=65504.0
L=pd.read_parquet("data/processed/labels/labels_master.parquet")
# stored correlations exist for all backbone edges
L=L.dropna(subset=["r_tumor","r_normal","delta_r","support_cancers"]).copy()
RT=L.r_tumor.values; RN=L.r_normal.values; d=(RN-RT); sup=L.support_cancers.values
# loosest lost/stable across the planned grid (STRONG>=0.4, WEAK<=0.4, DR>=0.2, support>=3)
loose_lost   = (sup>=3)&(RN>=0.4)&(RT<0.4)&(d>=0.2)
loose_stable = (sup>=3)&(np.abs(L.delta_r.values)<0.1)&(np.minimum(RT,RN)>=0.4)
keep = loose_lost | loose_stable
U=L[keep][["gene_a","gene_b","r_tumor","r_normal","delta_r","support_cancers"]].reset_index(drop=True)
print(f"union of labelable edges across grid: {len(U)}",flush=True)

# per-gene feature matrix + family map (for bio_only selection)
GM=pd.read_parquet("data/processed/features/gene_matrix.parquet").set_index("entrez")
gidx={int(e):i for i,e in enumerate(GM.index)}
X=np.nan_to_num(GM.to_numpy(np.float32),posinf=F16MAX,neginf=-F16MAX); D=X.shape[1]
gcols=list(GM.columns)
def fam(c):
    if c.startswith("e") and c[1:].isdigit(): return "ESM"
    if c.startswith("go"): return "GO"
    if c.startswith("pfam"): return "Pfam"
    if c in ("degree","log_degree","mean_string","mean_nsrc"): return "Network"
    if c.startswith("mut"): return "Mutation"
    if c.startswith("cnv"): return "CNV"
    return "Other"
gene_fam=[fam(c) for c in gcols]
mech_dims=[i for i,f in enumerate(gene_fam) if f in ("Network","Mutation","CNV")]  # the 44-dim bio_only source

ia=np.array([gidx[int(g)] for g in U.gene_a]); ib=np.array([gidx[int(g)] for g in U.gene_b])
A=X[ia]; B=X[ib]
blk=np.concatenate([np.minimum(A,B),np.maximum(A,B),np.abs(A-B),A*B],axis=1).astype(np.float16)
names=[f"{op}_{i}" for op in ("min","max","absdiff","prod") for i in range(D)]
df=pd.DataFrame(blk,columns=names)
for c in U.columns: df[c]=U[c].values
os.makedirs("data/processed/pairs",exist_ok=True)
df.to_parquet("data/processed/pairs/sweep_union.parquet",index=False)
# bio_only pair-feature indices = the 4 operators over the mechanistic gene-dims
bio_cols=[f"{op}_{i}" for op in ("min","max","absdiff","prod") for i in mech_dims]
json.dump({"D":D,"mech_dims":mech_dims,"bio_cols":bio_cols,"n_edges":len(U)},
          open("data/processed/pairs/sweep_union_meta.json","w"))
print(f"wrote sweep_union.parquet ({df.shape}) ; bio_only cols={len(bio_cols)}",flush=True)
