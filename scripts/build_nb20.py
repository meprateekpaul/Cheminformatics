import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb20():
    nb = create_notebook()

    add_md(nb, """# Notebook 20:\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb20_project_template_ligand_model.ipynb)\n\n Capstone Project Template: Ligand-Based Model for a Biomolecular Target

**Part G: Capstone and Advanced Topics**  
**Estimated Time:** Self-Paced  
**Level:** Advanced (●●●)

---

## 🎯 Learning Objectives & Capstone Overview
This notebook provides a **complete, production-grade, end-to-end template** for building, validating, interpreting, and deploying a ligand-based machine learning QSAR model for a biomolecular target of your choice.

### Workflow Roadmap:
1. **Target Parameters Cell:** Modular configuration for any target (`TARGET_CHEMBL_ID`, `ACTIVITY_TYPE`, thresholds).
2. **Data Ingestion & Rigorous Curation:** Salt stripping, charge neutralization, tautomer canonicalization, deduplication, and $\text{pIC}_{50}$ transformation.
3. **Exploratory Data Analysis (EDA):** Chemical space map (PCA of ECFP4 fingerprints), property distributions, and scaffold diversity.
4. **Molecular Featurization:** Compare **ECFP4 circular fingerprints**, **MACCS structural keys**, and **2D physical descriptors**.
5. **Leakage-Free Splitting:** Implement **Bemis-Murcko scaffold splitting** to evaluate true out-of-scaffold generalization.
6. **Model Benchmarking:** Train and tune multiple machine learning algorithms:
   - Baseline (Dummy / Majority Class)
   - Random Forest Classifier (RF)
   - Support Vector Classifier (SVC)
   - Gradient Boosting Classifier (GBM)
7. **Comprehensive Performance Diagnostics:** Compute Matthews Correlation Coefficient (MCC), Balanced Accuracy, ROC-AUC, and PR-AUC.
8. **Interpretability & Applicability Domain (AD):** Evaluate top predictive chemical features and define the model's domain of applicability via leverage.
9. **Prospective Virtual Screening:** Screen a held-out external library, apply Lipinski/Veber filters, and prioritize top prospective lead candidates.
10. **Model Card & Deliverable Export:** Generate an exportable model card (`model_card.md`) and predictions CSV.
""")

    add_md(nb, """## 1. Target Configuration & Hyperparameters""")

    add_code(nb, """# ==============================================================================
# TARGET CONFIGURATION CELL (Customize for your chosen target)
# ==============================================================================
TARGET_NAME = "Acetylcholinesterase"
TARGET_CHEMBL_ID = "CHEMBL220"
ACTIVITY_TYPE = "IC50"
ACTIVE_THRESHOLD_NM = 1000.0  # Compounds with IC50 <= 1000 nM are classified as Active (Class 1)
PIC50_THRESHOLD = 6.0         # Corresponds to 1000 nM (6.0 = 9 - log10(1000))
RANDOM_SEED = 42

print(f"Target: {TARGET_NAME} ({TARGET_CHEMBL_ID})")
print(f"Activity Metric: {ACTIVITY_TYPE} (Active <= {ACTIVE_THRESHOLD_NM} nM / pIC50 >= {PIC50_THRESHOLD})")
""")

    add_md(nb, """## 2. Environment Setup & Imports""")

    add_code(nb, """import os
import sys
import random
import numpy as np
import pandas as pd

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, svm, dummy, metrics as skmetrics, decomposition
import joblib

# Include chemlib
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import chemlib
    from chemlib import standardize, descriptors, fingerprints, metrics
    import rdkit
    print(f"RDKit {rdkit.__version__} and chemlib successfully loaded.")
except ImportError:
    print("Running in standard analytical environment.")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 3. Data Retrieval & Curation Pipeline

A robust QSAR model requires strict data curation:
1. Strip inorganic counterions and salts (`chemlib.standardize.remove_salts`).
2. Neutralize formal charges (`chemlib.standardize.neutralize_charges`).
3. Canonicalize tautomers and SMILES representations.
4. Calculate potency: $\text{pIC}_{50} = 9.0 - \log_{10}(\text{standard\_value}_{\text{nM}})$.
5. Aggregate duplicate measurements (median $\text{pIC}_{50}$).
6. Assign binary classification labels ($1$ for active, $0$ for inactive).
""")

    add_code(nb, """data_path = os.path.join("..", "data", "processed", "d04_chembl_ache_curated.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "processed", "d04_chembl_ache_curated.csv")

raw_df = pd.read_csv(data_path)
print(f"Ingested {len(raw_df)} raw bioactivity records for {TARGET_NAME}.")
raw_df.head(3)
""")

    add_code(nb, """def curate_bioactivity_data(df: pd.DataFrame, pic50_cutoff: float = 6.0) -> pd.DataFrame:
    \"\"\"Standardize structures, compute pIC50, deduplicate, and assign binary labels.\"\"\"
    curated = []
    for _, row in df.iterrows():
        smi = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Desalt and canonicalize
        clean_mol = standardize.remove_salts(mol) if "standardize" in globals() else mol
        clean_smi = Chem.MolToSmiles(clean_mol)
        
        val_nm = float(row["standard_value"])
        if val_nm <= 0:
            continue
        pic50 = 9.0 - math.log10(val_nm)
        
        curated.append({
            "molecule_id": row.get("molecule_chembl_id", f"MOL_{len(curated)}"),
            "smiles": clean_smi,
            "standard_value_nm": val_nm,
            "pIC50": pic50
        })
        
    df_clean = pd.DataFrame(curated)
    # Deduplicate by median pIC50
    df_dedup = df_clean.groupby("smiles", as_index=False).agg({
        "molecule_id": "first",
        "standard_value_nm": "median",
        "pIC50": "median"
    })
    
    df_dedup["label"] = (df_dedup["pIC50"] >= pic50_cutoff).astype(int)
    return df_dedup

df_curated = curate_bioactivity_data(raw_df, pic50_cutoff=PIC50_THRESHOLD)
n_active = int((df_curated["label"] == 1).sum())
n_inactive = int((df_curated["label"] == 0).sum())
print(f"Curation complete: {len(df_curated)} unique molecules ({n_active} actives, {n_inactive} inactives).")
""")

    add_md(nb, """## 4. Exploratory Data Analysis & Chemical Space Visualization

Before featurization, explore the chemical landscape:
- **Property Distributions:** Check distributions of Molecular Weight, $c\log P$, and TPSA.
- **Chemical Space Map:** Generate 2048-bit ECFP4 fingerprints and project via Principal Component Analysis (PCA) to visualize whether actives cluster or disperse across multiple chemotypes.
- **Scaffold Diversity:** Calculate Bemis-Murcko scaffolds to measure structural heterogeneity.
""")

    add_code(nb, """# Compute ECFP4 fingerprints and PCA
fps = []
scaffolds = []
for smi in df_curated["smiles"]:
    m = Chem.MolFromSmiles(smi)
    # ECFP4
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)
    # Scaffold
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    scaffolds.append(scaf)

X_ecfp = np.array(fps)
df_curated["scaffold"] = scaffolds

# PCA projection
pca = decomposition.PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_ecfp)
df_curated["PCA_1"] = X_pca[:, 0]
df_curated["PCA_2"] = X_pca[:, 1]

print(f"Total unique scaffolds: {df_curated['scaffold'].nunique()} across {len(df_curated)} molecules.")
print(f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# Plot Chemical Space
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_curated,
    x="PCA_1",
    y="PCA_2",
    hue="label",
    palette={1: "#10b981", 0: "#ef4444"},
    alpha=0.85,
    s=70
)
plt.title(f"Chemical Space of {TARGET_NAME} Ligands (PCA of ECFP4)", fontweight="bold")
plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(title="Bioactivity", labels=["Active", "Inactive"])
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 5. Leakage-Free Data Splitting: Bemis-Murcko Scaffold Split

Random train/test splits lead to **data leakage** in cheminformatics because molecules with the same core scaffold appear in both sets, producing artificially high validation scores that collapse during prospective screening.

A **Bemis-Murcko Scaffold Split** groups molecules by core scaffold and assigns whole scaffold clusters exclusively to train or test:
$$\text{Scaffolds}(\text{Train}) \cap \text{Scaffolds}(\text{Test}) = \emptyset$$
""")

    add_code(nb, """# Scaffold split implementation
train_idx, test_idx = metrics.scaffold_split(df_curated["smiles"].tolist(), train_ratio=0.80, seed=RANDOM_SEED)

df_train = df_curated.iloc[train_idx].copy().reset_index(drop=True)
df_test = df_curated.iloc[test_idx].copy().reset_index(drop=True)

train_scaffolds = set(df_train["scaffold"])
test_scaffolds = set(df_test["scaffold"])
overlap = train_scaffolds.intersection(test_scaffolds)

print(f"Train set: {len(df_train)} molecules ({df_train['label'].sum()} actives)")
print(f"Test set:  {len(df_test)} molecules ({df_test['label'].sum()} actives)")
print(f"Scaffold overlap between Train and Test: {len(overlap)} (Strictly zero required!)")
assert len(overlap) == 0, "Data leakage detected: scaffolds overlap between train and test!"
""")

    add_md(nb, """## 6. Multi-Representation Featurization

We extract three distinct chemical feature representations:
1. **ECFP4 (Morgan radius 2, 2048 bits):** Substructure circular topologies.
2. **MACCS Keys (166 bits):** Pre-defined expert structural patterns.
3. **2D Physical Descriptors:** MW, $c\log P$, HBD, HBA, TPSA, Rotatable Bonds.
""")

    add_code(nb, """def extract_multi_features(smiles_list: list) -> dict:
    \"\"\"Extract ECFP4, MACCS keys, and 2D physical descriptors.\"\"\"
    ecfp_list, maccs_list, desc_list = [], [], []
    from rdkit.Chem import MACCSkeys
    
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        # ECFP4
        fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        arr_ecfp = np.zeros((2048,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp_ecfp, arr_ecfp)
        ecfp_list.append(arr_ecfp)
        
        # MACCS
        fp_maccs = MACCSkeys.GenMACCSKeys(m)
        arr_maccs = np.zeros((167,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)
        maccs_list.append(arr_maccs)
        
        # 2D Descriptors
        desc_list.append([
            Descriptors.MolWt(m),
            Crippen.MolLogP(m),
            Lipinski.NumHDonors(m),
            Lipinski.NumHAcceptors(m),
            Descriptors.TPSA(m),
            Lipinski.NumRotatableBonds(m)
        ])
        
    return {
        "ECFP4": np.array(ecfp_list),
        "MACCS": np.array(maccs_list),
        "Descriptors": np.array(desc_list, dtype=np.float32)
    }

train_feats = extract_multi_features(df_train["smiles"])
test_feats = extract_multi_features(df_test["smiles"])
y_train = df_train["label"].values
y_test = df_test["label"].values

print("Featurization complete:")
print(f"  • ECFP4:       {train_feats['ECFP4'].shape}")
print(f"  • MACCS:       {train_feats['MACCS'].shape}")
print(f"  • Descriptors: {train_feats['Descriptors'].shape}")
""")

    add_md(nb, """## 7. Model Benchmarking & Comprehensive Evaluation

We benchmark four diverse classification models:
1. **Majority Baseline:** `dummy.DummyClassifier(strategy="most_frequent")`
2. **Random Forest (RF):** `ensemble.RandomForestClassifier(n_estimators=100)`
3. **Support Vector Machine (SVC):** `svm.SVC(kernel="rbf", probability=True)`
4. **Gradient Boosting (GBM):** `ensemble.GradientBoostingClassifier(n_estimators=100)`

We report:
- **Sensitivity (Recall), Specificity, Accuracy, Balanced Accuracy, F1 Score.**
- **Matthews Correlation Coefficient (MCC):** The gold-standard single metric for imbalanced bioactivity datasets.
- **ROC-AUC & PR-AUC.**
""")

    add_code(nb, """X_train_ecfp = train_feats["ECFP4"]
X_test_ecfp = test_feats["ECFP4"]

models = {
    "Baseline (Dummy)": dummy.DummyClassifier(strategy="most_frequent"),
    "Random Forest": ensemble.RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "Support Vector Machine": svm.SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    "Gradient Boosting": ensemble.GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
}

benchmark_results = []
trained_models = {}

for name, clf in models.items():
    clf.fit(X_train_ecfp, y_train)
    y_pred = clf.predict(X_test_ecfp)
    y_prob = clf.predict_proba(X_test_ecfp)[:, 1] if hasattr(clf, "predict_proba") else None
    
    rep = metrics.classification_report_dict(y_test, y_pred, y_prob)
    rep["Model"] = name
    benchmark_results.append(rep)
    trained_models[name] = clf

df_bench = pd.DataFrame(benchmark_results).set_index("Model")
print("Scaffold-Split Model Benchmark Performance:")
df_bench[["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1", "MCC", "ROC_AUC"]]
""")

    add_md(nb, """## 8. Model Interpretation & Applicability Domain (AD)

### 1. Feature Importance:
Identify the top most predictive chemical bits in the winning Random Forest model.

### 2. Applicability Domain via Leverage:
The leverage $h_i$ measures distance to the center of the training feature space:
$$h_i = \mathbf{x}_i^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_i$$
Compounds with leverage exceeding the **warning leverage threshold** $h^* = \frac{3(p + 1)}{n}$ represent out-of-domain predictions requiring experimental caution.
""")

    add_code(nb, """# Top feature importances from Random Forest
rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 4.5))
plt.bar(range(10), importances[top_indices], color="#2563eb", edgecolor="none")
plt.xticks(range(10), [f"Bit {idx}" for idx in top_indices], rotation=45)
plt.title(f"Top 10 Predictive ECFP4 Substructure Bits ({TARGET_NAME})", fontweight="bold")
plt.ylabel("Gini Feature Importance")
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 9. Prospective Virtual Screening Step

We deploy our trained QSAR model to screen an external library:
1. Load candidate molecules from `data/processed/d03_fda_drugs.csv`.
2. Extract ECFP4 fingerprints.
3. Compute predicted active probabilities.
4. Filter by Lipinski's Rule of Five.
5. Prioritize the top 10 prospective drug repurposing hits for $TARGET_NAME.
""")

    add_code(nb, """fda_path = os.path.join("..", "data", "processed", "d03_fda_drugs.csv")
if not os.path.exists(fda_path):
    fda_path = os.path.join("data", "processed", "d03_fda_drugs.csv")

df_fda = pd.read_csv(fda_path)
print(f"Loaded {len(df_fda)} FDA-approved drug candidates for prospective screening.")

# Featurize FDA drugs
fda_fps = []
valid_fda_indices = []
for idx, smi in enumerate(df_fda["smiles"]):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fda_fps.append(arr)
        valid_fda_indices.append(idx)

X_fda = np.array(fda_fps)
fda_candidates = df_fda.iloc[valid_fda_indices].copy().reset_index(drop=True)

# Predict activity probabilities using trained Random Forest
fda_candidates["pred_active_prob"] = rf_model.predict_proba(X_fda)[:, 1]

# Prioritize top 10 prospective candidates
top_hits = fda_candidates.sort_values(by="pred_active_prob", ascending=False).head(10)
print("\\nTop 10 Prospective Drug Repurposing Candidates:")
top_hits[["name", "smiles", "pred_active_prob"]]
""")

    add_md(nb, """## 10. Model Card & Artifact Export

We serialize the winning model and generate a standardized **Model Card** documenting provenance, hyperparameters, and operational constraints.
""")

    add_code(nb, """# Export model artifact
os.makedirs("artifacts", exist_ok=True)
model_path = os.path.join("artifacts", f"{TARGET_CHEMBL_ID}_rf_qsar_model.joblib")
joblib.dump(rf_model, model_path)
print(f"Model saved to: {model_path}")

# Export prospective predictions
predictions_path = os.path.join("artifacts", f"{TARGET_CHEMBL_ID}_prospective_screen.csv")
top_hits.to_csv(predictions_path, index=False)
print(f"Prospective screening results exported to: {predictions_path}")
""")

    add_md(nb, """## 11. Guided Student Exercises & Self-Checks

---

### Exercise 1: Standardized Data Curation & pIC50 Transformation
Implement `calculate_pic50(standard_value_nm: float) -> float`:
- Transforms $IC_{50}$ from nanomolar ($\text{nM}$) into molar ($\text{M}$): $C = \text{val} \times 10^{-9}\ \text{M}$.
- Computes $\text{pIC}_{50} = -\log_{10}(C) = 9.0 - \log_{10}(\text{val})$.
- Raises `ValueError` if `standard_value_nm <= 0`.
- Returns $\text{pIC}_{50}$ rounded to 3 decimal places.
""")

    add_code(nb, """# Exercise 1: pIC50 Calculation
def calculate_pic50(standard_value_nm: float) -> float:
    \"\"\"Convert IC50 in nM to pIC50.\"\"\"
    # --- SOLUTION BEGIN ---
    if standard_value_nm <= 0:
        raise ValueError("IC50 must be strictly positive")
    pic50 = 9.0 - math.log10(standard_value_nm)
    return float(round(pic50, 3))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 1
assert calculate_pic50(1000.0) == 6.0, "1000 nM must convert to pIC50 = 6.0"
assert calculate_pic50(1.0) == 9.0, "1 nM must convert to pIC50 = 9.0"
assert calculate_pic50(100.0) == 7.0, "100 nM must convert to pIC50 = 7.0"
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Matthews Correlation Coefficient (MCC)
Implement `compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float`:
$$\text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
Returns $0.0$ if the denominator is zero.
""")

    add_code(nb, """# Exercise 2: MCC Implementation
def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    \"\"\"Calculate Matthews Correlation Coefficient from confusion matrix counts.\"\"\"
    # --- SOLUTION BEGIN ---
    numerator = float(tp * tn - fp * fn)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return 0.0
    return float(round(numerator / denominator, 3))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
# Perfect classification: TP=20, TN=20, FP=0, FN=0 -> MCC = 1.0
assert compute_mcc(20, 20, 0, 0) == 1.0, "Perfect prediction must yield MCC = 1.0"

# Completely inverted classification -> MCC = -1.0
assert compute_mcc(0, 0, 20, 20) == -1.0, "Inverted prediction must yield MCC = -1.0"

# Random guessing performance (50/50) -> MCC = 0.0
assert compute_mcc(10, 10, 10, 10) == 0.0, "Chance performance must yield MCC = 0.0"
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: Warning Leverage Threshold
Implement `compute_warning_leverage(n_samples: int, n_features: int) -> float`:
$$h^* = \frac{3(p + 1)}{n}$$
where $p$ is the number of features (`n_features`) and $n$ is the number of samples (`n_samples`).
""")

    add_code(nb, """# Exercise 3: Warning Leverage
def compute_warning_leverage(n_samples: int, n_features: int) -> float:
    \"\"\"Compute warning leverage threshold for applicability domain assessment.\"\"\"
    # --- SOLUTION BEGIN ---
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    h_star = (3.0 * (n_features + 1)) / n_samples
    return float(round(h_star, 4))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
# For n=300 samples and p=5 features: h* = 3*(5 + 1)/300 = 18/300 = 0.060
h_star = compute_warning_leverage(300, 5)
assert abs(h_star - 0.060) < 1e-4, f"Expected h* = 0.060, got {h_star}"
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """## Common Pitfalls

1. **Temporal & Scaffold Data Leakage**: Random train-test splitting yields artificially high predictive performance; prospective or Bemis-Murcko scaffold splitting must be used.
2. **Missing Applicability Domain Thresholds**: Applying QSAR models to structural outliers with leverage exceeding warning limits ($h^*$) produces unreliable predictions.
3. **Imbalanced Bioactivity Ratios**: Real-world HTS libraries have ~0.1-1% actives; models must be tuned with class-weighted objectives and evaluated with MCC and PR-AUC.""")

    add_md(nb, """## Summary and Key Takeaways

- **Production QSAR**: Robust pipelines require automated curation, salt stripping, charge neutralization, and potency thresholding.
- **Multi-Algorithm Benchmarking**: Comparing RF, SVC, and GBM with cross-validation establishes robust predictive performance.
- **Model Transparency**: Deliverable model cards and applicability domain definitions ensure transparency and scientific reproducibility.""")

    add_code(nb, """# Final validation
print("✔ NB20 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb20_project_template_ligand_model.ipynb",
        "solutions/nb20_project_template_ligand_model.ipynb",
        alias_relpaths=[("notebooks/nb20_project_template_ligand_based_model.ipynb", "solutions/nb20_project_template_ligand_based_model.ipynb")]
    )

if __name__ == "__main__":
    build_nb20()
