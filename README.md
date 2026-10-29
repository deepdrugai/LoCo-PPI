# Co-expression loss prediction in the cancer interactome


An interaction is labeled lost when its two proteins are co-expressed in normal tissue but lose that
correlation in tumor, pooled across the 16 TCGA cancer types with enough adjacent-normal samples.
We predict these labels from sequence, domain, ontology, mutation and copy-number features, and
score the model under gene-disjoint cross-validation, where both proteins of a test pair are held
out of training. NECARE is run under the same split on its own task for comparison.

## Install

Python 3.11.

    pip install -r requirements.txt

## Data

Download the data archive from Zenodo (DOI below) into data/. It holds the per-gene feature matrix,
the labels and the interactome edges. The per-pair feature matrices are not shipped; they are
rebuilt from the gene matrix by the pairing step below. Original sources are public: BioGRID, STRING,
HIPPIE, TCGA (via the GDC), NECARE, LINCS L1000, Reactome, Gene Ontology, UniProt, and the Broad
Drug Repurposing Hub.

    Zenodo: <DOI>

## Scripts and order to run

    src/features/assemble_pairs.py   build the labeled pair matrix from the gene matrix
    src/models/sweep_build_union.py  build the threshold-sweep pair matrix
    src/models/train.py              gene-disjoint cross-validation of the lost and gained models
    src/models/ablate.py             feature-family ablation
    src/models/baselines.py          logistic-regression and degree-only baselines
    src/models/graphsage.py          inductive GraphSAGE baseline
    src/models/export_model.py       fit the final models on all labeled pairs and save them
    src/models/benchmark_xgb.py      our model on NECARE's curated task, matched gene-disjoint folds
    src/models/mcnemar_compare.py    paired comparison with NECARE (McNemar's test)

Run from the directory that holds data/ and results/. Build the pairs first, then train and evaluate.

## Models

model/ holds the trained XGBoost models in native format: model_A_lost.ubj (lost vs stable) and
model_B_gained.ubj (gained vs stable), each with a .features.txt listing the feature order.
Reproduce them with python src/models/export_model.py.

## NECARE comparison

The comparison also needs the NECARE code, adapted to DGL 2.3; it is kept separately and its
predictions are written to results/necare/ before mcnemar_compare.py is run.

