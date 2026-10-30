# Dataset Registry (D01 – D18)

This directory documents the provenance, licenses, and checksums for datasets used throughout the course.

| Dataset ID | Content | Source | Target File | Used In |
|---|---|---|---|---|
| **D01** | Slide SMILES lists (10 draw, 6 write) | Lecture Decks 01 & 02 | `data/processed/d01_slide_draw_smiles.csv`, `d01_slide_write_smiles.csv` | `nb03` |
| **D02** | Alkane isomers & experimental boiling points | NIST Chemistry WebBook | `data/processed/d02_alkanes_bp.csv` | `nb01`, `nb08` |
| **D03** | Curated FDA-approved drug subset with 3D coordinates | PubChem / DrugBank open subset | `data/processed/d03_fda_drugs.sdf`, `d03_fda_drugs.csv` | `nb05`, `nb07` |
| **D04** | ChEMBL human AChE bioactivity (CHEMBL220) | ChEMBL DB | `data/processed/d04_chembl_ache_curated.csv` | `nb06`, `nb10`, `nb13` |
| **D05** | ChEMBL BACE-1 bioactivity (CHEMBL4822) | ChEMBL DB | `data/processed/d05_chembl_bace1.csv` | `nb08`, `nb13`, `nb18` |
| **D06** | ChEMBL MAO-B Ki bioactivity (CHEMBL2039) | ChEMBL DB | `data/processed/d06_chembl_maob.csv` | `nb13` |
| **D07** | P-gp substrates / inhibitors classification dataset | TDC / ChEMBL | `data/processed/d07_pgp_classification.csv` | `nb07`, `nb13` |
| **D08** | YMDB metabolite structure file sample | YMDB / HMDB | `data/processed/d08_ymdb_sample.csv` | `nb05`, `nb06` |
| **D09** | ZINC fragment & drug-like subsets | ZINC20 / ZINC15 | `data/processed/d09_zinc_subset.csv` | `nb10`, `nb15`, `nb19`, `A1` |
| **D10** | PDB receptor complexes (AChE 1EVE / BACE-1 1FKN) | RCSB Protein Data Bank | `data/processed/d10_1eve_receptor.pdbqt` | `nb16`, `nb18` |
| **D11** | AlphaFold model for target pLDDT demonstration | AlphaFold DB | `data/processed/d11_alphafold_sample.cif` | `nb16` |
| **D12** | Differential gene expression table (RNA-seq / limma) | Synthetic / GEO benchmark | `data/processed/d12_differential_gene_expression.csv` | `nb16` |
| **D13** | Hansch-type substituent series (logP, sigma, log 1/C) | Hansch & Leo (1979) / Literature | `data/processed/d13_hansch_series.csv` | `nb12` |
| **D14** | Free-Wilson substituent matrix (R1, R2, activity) | Free & Wilson (1964) | `data/processed/d14_free_wilson_series.csv` | `nb12` |
| **D15** | Conformational torsion profiles (O-C-C-O, C-C-C-C) | CSD survey / MMFF scans | `data/processed/d15_torsion_profiles.csv` | `nb06` |
| **D16** | ADME & Tox benchmarking sets (ESOL, FreeSolv, Ames) | MoleculeNet | `data/processed/d16_esol_solubility.csv` | `nb15`, `nb21` |
| **D17** | Decoy library (property-matched DUD-E style) | Generated via chemlib | `data/processed/d17_ache_decoys.csv` | `nb10`, `nb18` |
| **D18** | Congeneric 3D-QSAR series | Steroid benchmark | `data/processed/d18_3d_qsar_congeneric.csv` | `nb14` |
