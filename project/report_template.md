# Capstone Project Report: [Title of Your Project]

**Author:** [Your Name]  
**Course:** Practical Cheminformatics & Computer-Aided Drug Design  
**Date:** [Date]  
**Target:** [Target Name, e.g., Acetylcholinesterase, CHEMBL220]  

---

## 1. Executive Summary
- Brief 150-word abstract summarizing the biological goal, data scale, best modeling approach, key validation metrics, and top prospective candidates discovered.

---

## 2. Biological Background & Target Significance
- Target structure, physiological function, and therapeutic relevance.
- Known binding pocket residues, catalytic mechanisms, and approved drug examples.

---

## 3. Dataset Curation & Exploratory Data Analysis
- Data retrieval query (ChEMBL ID, assay types, units).
- Curation steps applied: salt removal, tautomer canonicalization, duplicate resolution.
- Chemical space visualization (PCA/t-SNE of ECFP4) and activity distribution.
- Scaffold diversity and Bemis-Murcko framework analysis.

---

## 4. Modeling Strategy & Validation
- Featurization methods compared (ECFP4, MACCS, Descriptors, Pharmacophore signatures).
- Machine learning architectures evaluated (e.g. Random Forest, SVM, LightGBM, MLP).
- Splitting strategy: rationale for Bemis-Murcko scaffold split vs. random split.
- Performance metrics table (MCC, ROC-AUC, PR-AUC, F1, Balanced Accuracy).
- Discussion of the chosen champion model.

---

## 5. Model Interpretation & Applicability Domain
- Top molecular features driving predictions (SHAP / Gini importance).
- Mapping of predictive bits back to 2D chemical substructures.
- Applicability domain analysis (Williams plot / leverage vs. standardized residuals).
- Analysis of failure modes: activity cliffs and false negatives.

---

## 6. Prospective Virtual Screening
- Screening results on external compound library (e.g. ZINC / natural products).
- Application of drug-likeness filters (Lipinski Rule of 5, Veber, PAINS elimination).
- Detailed profile of top 5 prospective leads with interaction hypotheses.

---

## 7. Conclusions & Next Steps
- Summary of learnings, methodological limitations, and suggested wet-lab or docking validations.

---

## 8. References
- Primary literature, software citations (RDKit, Scikit-learn, etc.), and database sources.
