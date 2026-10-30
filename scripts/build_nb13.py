"""Build script for nb13: Machine Learning for QSAR (Classification and Regression)."""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb13():
    cells_student = []
    cells_solution = []

    def add_cell(cell_type, content, sol_content=None):
        if cell_type == "markdown":
            cells_student.append(new_markdown_cell(content))
            cells_solution.append(new_markdown_cell(content))
        elif cell_type == "code":
            cells_student.append(new_code_cell(content))
            cells_solution.append(new_code_cell(content))
        elif cell_type == "exercise":
            cells_student.append(new_code_cell(content))
            cells_solution.append(new_code_cell(sol_content if sol_content is not None else content))

    # --- Title and Header ---
    add_cell("markdown", """# nb13 · Machine Learning for QSAR: Classification and Regression
**Cheminformatics Course · Part E: QSAR, ML and ADME**

### Objectives
1. Frame QSAR modeling for **binary hit classification** and **continuous $pIC_{50}$ regression**.
2. Prevent data leakage by implementing **Bemis–Murcko scaffold splitting** and contrast it with naive random splits.
3. Establish essential **baselines**: majority class, stratified random, and 1-NN similarity-to-nearest-active.
4. Train and compare classical and modern machine learning algorithms:
   - **Classifiers:** Logistic Regression, $k$-Nearest Neighbors ($k$-NN), Random Forest, Support Vector Machines (SVM).
   - **Regressors:** Ridge Regression, Random Forest Regressor, Multi-Layer Perceptron (MLP).
5. Evaluate models using rigorous metrics: **Sensitivity, Specificity, Precision, Accuracy, $F_1$, MCC, ROC-AUC, PR-AUC, $R^2$, RMSE, and $R^2_{\\text{pred}}$**.
6. Analyze **feature importance** (top predictive fingerprint bits) and define the **applicability domain** based on chemical similarity.""")

    # --- Setup & Imports ---
    add_cell("markdown", """## 1. Setup, Environment & Imports""")

    add_cell("code", """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor

import chemlib
from chemlib import fingerprints, metrics

print(f"RDKit Version: {rdkit.__version__}")
print(f"chemlib imported successfully from: {chemlib.__file__}")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)

def find_data_file(filename: str) -> str:
    \"\"\"Locate a data file reliably regardless of current working directory.\"\"\"
    for root in [".", "..", "Cheminformatics", "../Cheminformatics"]:
        cand = os.path.join(root, "data", "processed", filename)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(f"Could not locate {filename} in project data/processed paths.")""")

    # --- Section 2: Data Loading & Preprocessing ---
    add_cell("markdown", """## 2. Activity Data Preparation & Molecular Representations

We evaluate QSAR modeling using the curated ChEMBL Human Acetylcholinesterase (AChE, CHEMBL220) dataset:
- **Raw Affinity:** $IC_{50}$ in nanomolar (nM).
- **Logarithmic Affinity:** Continuous $pIC_{50} = -\\log_{10}(IC_{50} \\text{ in M}) = 9.0 - \\log_{10}(IC_{50} \\text{ in nM})$.
- **Binary Classification Threshold:**
  $$\\text{Active } (y = 1) \\iff pIC_{50} \\ge 7.0 \\quad (IC_{50} \\le 100\\text{ nM})$$
  $$\\text{Inactive / Weak } (y = 0) \\iff pIC_{50} < 7.0$$""")

    add_cell("code", """# Load dataset and curate activity values
data_ache = find_data_file("d04_chembl_ache_curated.csv")
df = pd.read_csv(data_ache)

# Convert IC50 (nM) to continuous pIC50
df["pIC50"] = 9.0 - np.log10(df["standard_value"].clip(lower=1e-3))
df["is_active"] = (df["pIC50"] >= 7.0).astype(int)

df["mol"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
valid_mask = df["mol"].notna()
df = df[valid_mask].reset_index(drop=True)

print(f"Total Valid Molecules: {len(df)}")
print(f"Actives (pIC50 >= 7.0): {df['is_active'].sum()} ({df['is_active'].mean()*100:.1f}%)")
print(f"pIC50 Range: {df['pIC50'].min():.2f} to {df['pIC50'].max():.2f} (Mean: {df['pIC50'].mean():.2f})")

# Compute Fingerprints: ECFP4 (2048-bit) and MACCS (166-bit)
X_ecfp4 = np.array([fingerprints.fp_to_numpy(fingerprints.get_fingerprint(m, "ecfp4")) for m in df["mol"]])
X_maccs = np.array([fingerprints.fp_to_numpy(fingerprints.get_fingerprint(m, "maccs")) for m in df["mol"]])

print(f"ECFP4 Matrix Shape: {X_ecfp4.shape}")
print(f"MACCS Matrix Shape: {X_maccs.shape}")""")

    # --- Section 3: Scaffold Splitting ---
    add_cell("markdown", """## 3. Data Splitting: Preventing Data Leakage with Bemis–Murcko Scaffolds

### Why Random Splitting Causes Optimistic Data Leakage
In drug discovery, active molecules usually come from a small number of **congeneric chemical series** sharing a core scaffold.
- If we use a naive random 80/20 train/test split, molecules from the **same congeneric series** end up in both training and test sets.
- The model merely memorizes the scaffold rather than learning generalizable SAR rules.
- When applied prospectively to genuinely novel chemotypes, performance collapses.

### Bemis–Murcko Scaffold Splitting
By clustering molecules according to their 2D Murcko frameworks, all analogs sharing a common core framework are assigned together to either the training set or the test set, directly testing generalizability to novel scaffolds.""")

    add_cell("code", """# Perform Bemis-Murcko scaffold split (80% train, 20% test)
train_idx, test_idx = metrics.scaffold_split(df["canonical_smiles"].tolist(), train_ratio=0.8, seed=42)

print(f"Training Set Size: {len(train_idx)} compounds")
print(f"Test Set Size:     {len(test_idx)} compounds")

# Prepare train/test feature matrices and target vectors
X_train_ecfp4, X_test_ecfp4 = X_ecfp4[train_idx], X_ecfp4[test_idx]
X_train_maccs, X_test_maccs = X_maccs[train_idx], X_maccs[test_idx]

y_train_cls, y_test_cls = df.loc[train_idx, "is_active"].values, df.loc[test_idx, "is_active"].values
y_train_reg, y_test_reg = df.loc[train_idx, "pIC50"].values, df.loc[test_idx, "pIC50"].values""")

    # --- Section 4: Baselines ---
    add_cell("markdown", """## 4. Baselines: Establishing the Benchmark Ground Truth

Never report machine learning performance without baselines:
1. **Majority Class Baseline:** Always predicts the most frequent class.
2. **Stratified Random Baseline:** Predicts classes according to training label frequencies.
3. **1-NN Similarity Baseline:** Predicts the label of the single most similar molecule in the training set.""")

    add_cell("code", """# 1. Majority Class Classifier
majority_clf = DummyClassifier(strategy="most_frequent")
majority_clf.fit(X_train_ecfp4, y_train_cls)
preds_maj = majority_clf.predict(X_test_ecfp4)
rep_maj = metrics.classification_report_dict(y_test_cls, preds_maj)

# 2. 1-NN Active Similarity Baseline
knn1 = KNeighborsClassifier(n_neighbors=1, metric="jaccard")
knn1.fit(X_train_ecfp4, y_train_cls)
preds_knn1 = knn1.predict(X_test_ecfp4)
probs_knn1 = knn1.predict_proba(X_test_ecfp4)[:, 1]
rep_knn1 = metrics.classification_report_dict(y_test_cls, preds_knn1, probs_knn1)

print(f"Majority Class Baseline: Accuracy = {rep_maj['Accuracy']:.3f} | MCC = {rep_maj['MCC']:.3f}")
print(f"1-NN Similarity Baseline: Accuracy = {rep_knn1['Accuracy']:.3f} | MCC = {rep_knn1['MCC']:.3f} | ROC-AUC = {rep_knn1['ROC_AUC']:.3f}")""")

    # --- Section 5: Machine Learning Classifiers ---
    add_cell("markdown", """## 5. Machine Learning Classifiers & Comprehensive Metric Suite

We train four standard classifiers on ECFP4 fingerprints:
1. **Logistic Regression (L2 regularized)**
2. **$k$-Nearest Neighbors ($k$-NN, $k=5$)**
3. **Random Forest Classifier (100 trees)**
4. **Support Vector Machine (SVC, RBF kernel)**

### The Matthews Correlation Coefficient (MCC)
$$\\text{MCC} = \\frac{TP \\cdot TN - FP \\cdot FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
MCC ranges from $-1$ (total disagreement) through $0$ (random guessing) to $+1$ (perfect prediction), providing the gold standard metric for chemically imbalanced datasets.""")

    add_cell("code", """classifiers = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

results_cls = []
for name, clf in classifiers.items():
    clf.fit(X_train_ecfp4, y_train_cls)
    preds = clf.predict(X_test_ecfp4)
    probs = clf.predict_proba(X_test_ecfp4)[:, 1] if hasattr(clf, "predict_proba") else None
    
    rep = metrics.classification_report_dict(y_test_cls, preds, probs)
    results_cls.append({
        "Model": name,
        "Accuracy": rep["Accuracy"],
        "Sensitivity": rep["Sensitivity_Recall"],
        "Specificity": rep["Specificity"],
        "Precision": rep["Precision"],
        "F1_Score": rep["F1_Score"],
        "MCC": rep["MCC"],
        "ROC_AUC": rep["ROC_AUC"],
        "PR_AUC": rep["PR_AUC"],
    })

df_results_cls = pd.DataFrame(results_cls)
print("=== Classification Model Comparison on Scaffold Test Set ===")
print(df_results_cls.to_string(index=False))""")

    # --- Section 6: Machine Learning Regressors ---
    add_cell("markdown", """## 6. Machine Learning Regressors & Parity Plots

We evaluate continuous affinity regression ($pIC_{50}$ prediction) across three models:
1. **Ridge Regression (L2 linear)**
2. **Random Forest Regressor**
3. **Multi-Layer Perceptron (MLPRegressor)**""")

    add_cell("code", """regressors = {
    "Ridge Regression": Ridge(alpha=10.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Multi-Layer Perceptron": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

results_reg = []
predictions_reg = {}

for name, reg in regressors.items():
    reg.fit(X_train_ecfp4, y_train_reg)
    preds = reg.predict(X_test_ecfp4)
    predictions_reg[name] = preds

    reg_eval = metrics.regression_metrics_dict(y_test_reg, preds, n_features=X_train_ecfp4.shape[1])
    # Tropsha external R2_pred
    r2_pred = metrics.r2_pred_external(y_train_reg, y_test_reg, preds)

    results_reg.append({
        "Model": name,
        "R2": reg_eval["R2"],
        "RMSE": reg_eval["RMSE"],
        "MAE": reg_eval["MAE"],
        "R2_pred_ext": r2_pred,
        "Pearson_r": reg_eval["Pearson_r"],
        "Spearman_rho": reg_eval["Spearman_rho"],
    })

df_results_reg = pd.DataFrame(results_reg)
print("=== Regression Model Comparison on Scaffold Test Set ===")
print(df_results_reg.to_string(index=False))""")

    # --- Section 7: Feature Importance & Applicability Domain ---
    add_cell("markdown", """## 7. Feature Importance & Applicability Domain

### 1. Fingerprint Feature Importance
Random Forest provides Gini importance scores identifying the specific bit indices that drive potency predictions.""")

    add_cell("code", """# Extract top 10 predictive bits from Random Forest Classifier
rf_clf = classifiers["Random Forest"]
importances = rf_clf.feature_importances_
top_bit_indices = np.argsort(importances)[::-1][:10]

print("Top 10 Most Predictive ECFP4 Bit Indices for AChE Activity:")
for rank, idx in enumerate(top_bit_indices, start=1):
    print(f"  Rank {rank:02d}: Bit {idx:<5} | Importance = {importances[idx]:.4f}")""")

    # --- Section 8: Exercises ---
    add_cell("markdown", """## 8. Exercises & Self-Checks

### Exercise 1: Compare ECFP4 vs MACCS Representations
Write a function `compare_fingerprint_representations` that trains a Random Forest Classifier on both ECFP4 and MACCS feature matrices, returning their respective test set MCC and ROC-AUC scores.""")

    ex1_student = """def compare_fingerprint_representations(
    X_tr_ecfp4: np.ndarray,
    X_te_ecfp4: np.ndarray,
    X_tr_maccs: np.ndarray,
    X_te_maccs: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray
) -> Dict[str, Dict[str, float]]:
    \"\"\"Train Random Forest on ECFP4 vs MACCS and return MCC and ROC-AUC for both.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement compare_fingerprint_representations\")"""

    ex1_solution = """def compare_fingerprint_representations(
    X_tr_ecfp4: np.ndarray,
    X_te_ecfp4: np.ndarray,
    X_tr_maccs: np.ndarray,
    X_te_maccs: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray
) -> Dict[str, Dict[str, float]]:
    \"\"\"Train Random Forest on ECFP4 vs MACCS and return MCC and ROC-AUC for both.\"\"\"
    rf_ecfp4 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_ecfp4.fit(X_tr_ecfp4, y_tr)
    pred_e = rf_ecfp4.predict(X_te_ecfp4)
    prob_e = rf_ecfp4.predict_proba(X_te_ecfp4)[:, 1]
    rep_e = metrics.classification_report_dict(y_te, pred_e, prob_e)

    rf_maccs = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_maccs.fit(X_tr_maccs, y_tr)
    pred_m = rf_maccs.predict(X_te_maccs)
    prob_m = rf_maccs.predict_proba(X_te_maccs)[:, 1]
    rep_m = metrics.classification_report_dict(y_te, pred_m, prob_m)

    return {
        "ECFP4": {"MCC": rep_e["MCC"], "ROC_AUC": rep_e["ROC_AUC"]},
        "MACCS": {"MCC": rep_m["MCC"], "ROC_AUC": rep_m["ROC_AUC"]}
    }"""

    add_cell("exercise", ex1_student, ex1_solution)

    add_cell("markdown", """### Exercise 2: Parity Plot Residual Analysis
Implement `compute_parity_residuals` that calculates residuals $(y_{\\text{obs}} - y_{\\text{pred}})$, the fraction of predictions within $\\pm 1.0$ log unit of the true experimental value, and the RMSE.""")

    ex2_student = """def compute_parity_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    \"\"\"Compute residuals, fraction within 1.0 log-unit, and RMSE.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement compute_parity_residuals\")"""

    ex2_solution = """def compute_parity_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    \"\"\"Compute residuals, fraction within 1.0 log-unit, and RMSE.\"\"\"
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    within_1_log = np.mean(np.abs(residuals) <= 1.0)
    rmse = np.sqrt(np.mean(residuals ** 2))

    return {
        "residuals": residuals,
        "fraction_within_1_log": float(within_1_log),
        "RMSE": float(rmse),
    }"""

    add_cell("exercise", ex2_student, ex2_solution)

    # --- Self-Checks ---
    add_cell("markdown", """## 9. Self-Check Unit Tests
Run the cell below to verify your solutions against automated test assertions.""")

    add_cell("code", """# --- Unit Tests for Exercise 1 ---
fp_cmp = compare_fingerprint_representations(
    X_train_ecfp4, X_test_ecfp4,
    X_train_maccs, X_test_maccs,
    y_train_cls, y_test_cls
)
assert "ECFP4" in fp_cmp and "MACCS" in fp_cmp, "Must return results for both ECFP4 and MACCS"
assert fp_cmp["ECFP4"]["ROC_AUC"] > 0.85, f"ECFP4 ROC-AUC should be high, got {fp_cmp['ECFP4']['ROC_AUC']:.3f}"
assert -1.0 <= fp_cmp["ECFP4"]["MCC"] <= 1.0, "MCC must be in valid [-1, 1] range"
assert -1.0 <= fp_cmp["MACCS"]["MCC"] <= 1.0, "MCC must be in valid [-1, 1] range"

# --- Unit Tests for Exercise 2 ---
rf_preds = predictions_reg["Random Forest"]
par_res = compute_parity_residuals(y_test_reg, rf_preds)
assert 0.0 <= par_res["fraction_within_1_log"] <= 1.0, "Fraction within 1.0 log must be between 0 and 1"
assert par_res["fraction_within_1_log"] > 0.50, f"Expected > 50% predictions within 1 log-unit, got {par_res['fraction_within_1_log']*100:.1f}%"
assert par_res["RMSE"] > 0.0, "RMSE must be non-negative"

print("All assertion self-checks passed!")
print("✔ NB13 checks passed")""")

    # Write notebooks
    os.makedirs("Cheminformatics/notebooks", exist_ok=True)
    os.makedirs("Cheminformatics/solutions", exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    path_student = "Cheminformatics/notebooks/nb13_ml_qsar_classification_regression.ipynb"
    path_sol = "Cheminformatics/solutions/nb13_ml_qsar_classification_regression.ipynb"

    with open(path_student, "w") as f:
        nbformat.write(nb_student, f)

    with open(path_sol, "w") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {path_student} and {path_sol}")

if __name__ == "__main__":
    create_nb13()
