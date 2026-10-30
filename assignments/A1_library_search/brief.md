# Assignment A1: Searching Chemical Libraries with a Template or Pharmacophore

**Points:** 15 Points (out of 30 for course assignments)  
**Deliverables:** Completed Jupyter notebook with code, outputs, and written commentary.

---

## 🎯 Objectives
1. Perform substructure/SMARTS queries against a chemical library.
2. Build 2D and 3D pharmacophore models from a set of known active ligands.
3. Compare ligand-based virtual screening performance using multiple similarity coefficients (Tanimoto, Dice, Cosine).
4. Evaluate retrieval accuracy using Receiver Operating Characteristic (ROC), PR curves, Enrichment Factor (EF1%), and BEDROC.
5. Identify and justify the top 10 diverse prospective hits.

---

## 📋 Tasks & Workflow
1. **Actives & Decoys Curation:** Load the provided set of known AChE/BACE-1 actives and property-matched decoys (`data/processed/d04_chembl_ache_curated.csv` and `d17_ache_decoys.csv`).
2. **SMARTS Substructure Filter:** Define a SMARTS pattern representing the core pharmacophoric motif (e.g. basic benzyl-piperidine or amino-tetrahydroacridine). Filter the library and record the hit rate.
3. **3-Point Pharmacophore Model:** Extract chemical features (HB Donor, HB Acceptor, Aromatic ring) using `chemlib.pharm`. Formulate a 3-point pharmacophore triangle with distance tolerances.
4. **Similarity Search Baseline:** Compute ECFP4 fingerprints and perform bulk similarity search using Tanimoto and Tversky coefficients.
5. **Screening Benchmark:** Generate ROC and PR curves. Calculate EF1% and BEDROC ($\alpha = 20$).
6. **Diversity Selection:** Cluster the top 50 hits using Bemis-Murcko scaffolds and select 10 structurally diverse hits for your purchase recommendation.

---

## ⚖️ Rubric
- **Data handling & standardization:** 20%
- **Search implementation (SMARTS + Similarity):** 25%
- **Pharmacophore design quality & rationale:** 20%
- **Evaluation rigor (ROC-AUC, EF1%, BEDROC):** 20%
- **Diversity selection & scientific write-up:** 15%
