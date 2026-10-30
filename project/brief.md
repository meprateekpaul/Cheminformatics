# Capstone Project Brief: Ligand-Based Machine Learning Model for a Biomolecular Target

## 🎯 Overview
In this capstone project, you will build, validate, interpret, and deploy a machine learning QSAR model for a clinically or biologically relevant biomolecular target of your choice.

Using the template in `notebooks/nb20_project_template_ligand_model.ipynb`, you will execute an end-to-end computational drug discovery workflow:
1. **Target Selection & Biological Context:** Choose a target in ChEMBL (e.g. human AChE, BACE-1, EGFR, hERG, JAK2, SARS-CoV-2 Mpro).
2. **Data Retrieval & Rigorous Curation:** Retrieve bioactivity data via API/cache, curate standard values, remove salts, uncharge, canonicalize, and aggregate duplicate measurements.
3. **Exploratory Data Analysis:** Visualize chemical space (PCA/t-SNE of ECFP4), examine activity distribution, and assess scaffold diversity.
4. **Molecular Featurization:** Compare at least three distinct feature representations (e.g. ECFP4 bit vectors, MACCS keys, 2D RDKit/Mordred physical descriptors, 2D pharmacophore signatures).
5. **Leakage-Free Modeling & Cross-Validation:** Split data using **Bemis-Murcko scaffold splitting** and nested cross-validation. Compare at least three algorithms (e.g. Random Forest, Support Vector Machines, Gradient Boosting, kNN) against naïve baselines.
6. **Model Interpretation & Applicability Domain:** Identify top predictive features (feature importance / SHAP) and evaluate the model's applicability domain (Williams plot / leverage).
7. **Prospective Virtual Screening:** Screen a held-out external library (e.g. ZINC fragment or drug-like subset), filter hits by drug-likeness (Lipinski/Veber), and propose top candidate molecules.

## 📦 Deliverables
1. **Executable Jupyter Notebook:** Parameterized and cleanly documented notebook based on `nb20`.
2. **Written Project Report:** 4–6 page report following `project/report_template.md`.
3. **Model Card:** Markdown summary documenting intended use, training data provenance, performance metrics, and limitations.
4. **Trained Model Artifact:** Serialized model file (`.joblib` or `.pkl`) and screened candidate predictions CSV.
