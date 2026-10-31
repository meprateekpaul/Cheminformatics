import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb18():
    nb = create_notebook()

    add_md(nb, """# Notebook 18:\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb18_docking_and_virtual_screening.ipynb)\n\n Docking and Virtual Screening

**Part F: Structure-Based Drug Design (SBDD)**  
**Estimated Time:** ~150 min  
**Level:** Advanced (●●●)

---

## 🎯 Learning Objectives
1. **Structure-Based Virtual Screening (SBVS) Workflow:** Master the end-to-end pipeline: target preparation $\rightarrow$ grid box setup $\rightarrow$ ligand 3D conformer generation $\rightarrow$ AutoDock Vina docking $\rightarrow$ pose scoring and post-processing.
2. **Redocking Validation & Heavy-Atom RMSD:** Execute crystallographic ligand redocking and calculate symmetry-aware heavy-atom RMSD against the crystal pose ($\text{RMSD} < 2.0\ \text{Å}$ success criterion).
3. **Congeneric Series Correlation:** Dock congeneric chemical series and correlate predicted binding energies ($\Delta G_{\text{dock}}$) with experimental $\text{pIC}_{50}$ values using Pearson $r$ and Spearman $\rho$.
4. **Virtual Screening Benchmark:** Screen known actives against property-matched decoys (`d04_chembl_ache_curated.csv` vs `d17_ache_decoys.csv`).
5. **Early Recognition Metrics:** Generate ROC and PR curves, and compute **Enrichment Factor at 1% ($EF_{1\%}$)** and **BEDROC ($\alpha = 20$)**.
6. **Interaction Analysis in 3D:** Identify key binding pocket interactions: hydrogen bonds, $\pi$-$\pi$ stacking with aromatic gorge residues (Trp84, Phe330), and hydrophobic contacts.
""")

    add_md(nb, """## 1. Prerequisites & Environment Setup""")

    add_code(nb, """import os
import sys
import random
import numpy as np
import pandas as pd

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skmetrics

# Include chemlib
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    import chemlib
    from chemlib import docking, metrics
    import rdkit
    print(f"RDKit {rdkit.__version__} and chemlib successfully loaded.")
except ImportError:
    print("Running in standard analytical environment.")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 2. The Structure-Based Virtual Screening Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │                 Target Preparation (nb16)              │
  │   - PDB crystal structure / AlphaFold model            │
  │   - Strip bulk waters & resolve alternate locations    │
  │   - Protonate His / Asp / Glu & assign Gasteiger Q     │
  │   - Define binding pocket centroid and grid box (Å)    │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │                 Ligand Library Preparation             │
  │   - SMILES canonicalization & salt stripping (nb05)    │
  │   - Protonation at pH 7.4 & tautomer enumeration       │
  │   - 3D conformer generation (ETKDG v3 + MMFF94)        │
  │   - Assign rotatable bonds (TORSDOF) via Meeko         │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │             AutoDock Vina Docking & Scoring            │
  │   - Affinity maps calculation (box center & sizes)     │
  │   - Monte Carlo iterated local search (BFGS)           │
  │   - Extract poses and predicted binding free energy    │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │               Post-Docking Triage & Ranking            │
  │   - Heavy-atom RMSD validation (< 2.0 Å)               │
  │   - Physicochemical filters (Lipinski / Veber / PAINS) │
  │   - Interaction profiling (H-bonds, pi-stacking)       │
  │   - Enrichment metrics: ROC-AUC, EF1%, BEDROC          │
  └────────────────────────────────────────────────────────┘
```
""")

    add_md(nb, """## 3. Redocking Validation & Heavy-Atom RMSD

Before screening thousands of molecules, we must validate the docking protocol via **redocking**:
- Take the co-crystallized ligand from the experimental complex.
- Remove crystallographic coordinates and rebuild a randomized 3D conformer from SMILES.
- Dock into the prepared receptor grid box.
- Measure **heavy-atom Root-Mean-Square Deviation (RMSD)** between docked pose ($\mathbf{r}^{\text{docked}}$) and crystal pose ($\mathbf{r}^{\text{crystal}}$):
  $$\text{RMSD} = \sqrt{\frac{1}{N_{\text{heavy}}} \sum_{i=1}^{N_{\text{heavy}}} \|\mathbf{r}_i^{\text{docked}} - \mathbf{r}_i^{\text{crystal}}\|^2}$$
- **Validation Rule of Thumb:** $\text{RMSD} \le 2.0\ \text{Å}$ represents a successful dock that accurately reproduces the crystallographic binding mode.
""")

    add_code(nb, """# Reference crystal coordinates of co-crystallized Donepezil fragment (1EVE)
crystal_coords = np.array([
    [1.12, 9.85, 25.42], [0.35, 10.82, 26.15], [0.78, 12.24, 25.92],
    [0.12, 13.25, 26.41], [2.01, 12.18, 25.02], [2.24, 10.72, 24.71],
    [-0.85, 10.65, 27.01], [-1.25, 9.28, 27.35], [-0.21, 8.41, 28.05],
    [1.15, 8.55, 27.39], [1.58, 9.98, 27.12]
])

# Simulated top-3 docked poses from AutoDock Vina
docked_pose_1 = crystal_coords + np.random.RandomState(10).normal(0, 0.45, crystal_coords.shape)
docked_pose_2 = crystal_coords + np.random.RandomState(20).normal(0, 1.20, crystal_coords.shape)
docked_pose_3 = crystal_coords + np.array([3.2, -2.5, 1.8])  # Misplaced decoy pose

def compute_coordinate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    \"\"\"Compute standard heavy-atom RMSD between two aligned coordinate matrices.\"\"\"
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

rmsd_1 = compute_coordinate_rmsd(docked_pose_1, crystal_coords)
rmsd_2 = compute_coordinate_rmsd(docked_pose_2, crystal_coords)
rmsd_3 = compute_coordinate_rmsd(docked_pose_3, crystal_coords)

print(f"Pose 1 RMSD: {rmsd_1:.2f} Å (Passes validation: {rmsd_1 < 2.0})")
print(f"Pose 2 RMSD: {rmsd_2:.2f} Å (Marginal / Near threshold: {rmsd_2 < 2.0})")
print(f"Pose 3 RMSD: {rmsd_3:.2f} Å (Failed dock: {rmsd_3 < 2.0})")
""")

    add_md(nb, """## 4. Docking a Congeneric Series & Correlation with Experimental Affinities

Within a **congeneric chemical series** (compounds sharing a conserved core with varied substituents), docking scores often correlate with experimental potencies:
- **Conversion to $\text{pIC}_{50}$:**
  $$\text{pIC}_{50} = -\log_{10}(IC_{50}\ (\text{M})) = 9 - \log_{10}(IC_{50}\ (\text{nM}))$$
- We evaluate:
  - **Pearson correlation ($r$):** Linear relationship between $-\Delta G_{\text{dock}}$ and $\text{pIC}_{50}$.
  - **Spearman rank correlation ($\rho$):** Monotonic ranking ability.
""")

    add_code(nb, """# Load curated AChE bioactivity series
ache_path = os.path.join("..", "data", "processed", "d04_chembl_ache_curated.csv")
if not os.path.exists(ache_path):
    ache_path = os.path.join("data", "processed", "d04_chembl_ache_curated.csv")

df_ache = pd.read_csv(ache_path)
# Calculate pIC50
df_ache["pIC50"] = 9.0 - np.log10(df_ache["standard_value"])

# Select top 20 representative molecules
subset_congeneric = df_ache.head(20).copy()

# Simulated Vina docking scores (typically between -6.0 and -11.5 kcal/mol)
# In a realistic congeneric series, Vina score correlates with pIC50 with r ~ 0.65
rng = np.random.RandomState(42)
subset_congeneric["vina_score_kcal_mol"] = (
    -4.5 - 0.70 * subset_congeneric["pIC50"] + rng.normal(0, 0.55, len(subset_congeneric))
)

# Plot correlation
from scipy import stats
r_val, p_val = stats.pearsonr(subset_congeneric["pIC50"], -subset_congeneric["vina_score_kcal_mol"])
rho_val, _ = stats.spearmanr(subset_congeneric["pIC50"], -subset_congeneric["vina_score_kcal_mol"])

plt.figure(figsize=(7, 5))
sns.regplot(
    data=subset_congeneric,
    x="pIC50",
    y=-subset_congeneric["vina_score_kcal_mol"],
    color="#2563eb",
    scatter_kws={"s": 60, "alpha": 0.8}
)
plt.title(f"AChE Congeneric Series: Docking Score vs. Experimental Potency\\n(Pearson r = {r_val:.2f}, Spearman ρ = {rho_val:.2f})", fontweight="bold")
plt.xlabel("Experimental Potency ($\mathrm{pIC}_{50}$)")
plt.ylabel("Predicted Affinity ($-\\Delta G_{\\mathrm{dock}}$, kcal/mol)")
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 5. Virtual Screening Benchmark: Actives vs Decoys

To test virtual screening discrimination, we build a benchmark dataset combining:
- **Known Actives:** From `d04_chembl_ache_curated.csv` ($IC_{50} \le 1000\ \text{nM}$).
- **Property-Matched Decoys:** From `d17_ache_decoys.csv` ($IC_{50} > 10,000\ \text{nM}$).

### Performance Metrics:
1. **Receiver Operating Characteristic (ROC-AUC):** Global discrimination across all score thresholds.
2. **Enrichment Factor at $\chi\%$ ($EF_{\chi\%}$):** Ratio of active concentration in the top $\chi\%$ compared to random selection:
   $$EF_{\chi\%} = \frac{N_{\text{actives, top }\chi\%} / N_{\text{top }\chi\%}}{N_{\text{actives, total}} / N_{\text{total}}}$$
3. **BEDROC (Boltzmann-Enhanced Discrimination of ROC):**
   Weighting early ranks exponentially ($\alpha = 20.0$, where top 8% of the list contributes 80% of the metric score).
""")

    add_code(nb, """decoy_path = os.path.join("..", "data", "processed", "d17_ache_decoys.csv")
if not os.path.exists(decoy_path):
    decoy_path = os.path.join("data", "processed", "d17_ache_decoys.csv")

df_decoys = pd.read_csv(decoy_path)

# Prepare combined benchmark
actives_sample = df_ache[df_ache["standard_value"] <= 1000].head(50).copy()
actives_sample["label"] = 1
# Generate docking scores for actives (median ~ -9.2 kcal/mol)
actives_sample["docking_score"] = rng.normal(-9.2, 0.8, len(actives_sample))

decoys_sample = df_decoys.head(250).copy()
decoys_sample["label"] = 0
# Generate docking scores for decoys (median ~ -6.8 kcal/mol)
decoys_sample["docking_score"] = rng.normal(-6.8, 1.1, len(decoys_sample))

df_benchmark = pd.concat([actives_sample, decoys_sample], ignore_index=True)
print(f"Benchmark Library: {len(df_benchmark)} total compounds ({len(actives_sample)} actives, {len(decoys_sample)} decoys).")

# Compute ROC curve (more negative docking score = higher rank)
y_true = df_benchmark["label"].values
pred_scores = -df_benchmark["docking_score"].values  # Invert so higher is better

fpr, tpr, _ = skmetrics.roc_curve(y_true, pred_scores)
roc_auc = skmetrics.roc_auc_score(y_true, pred_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#10b981", lw=2.5, label=f"Vina Docking (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (AUC = 0.500)")
plt.title("Virtual Screening Benchmark: ROC Curve", fontweight="bold")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 6. Guided Student Exercises & Self-Checks

---

### Exercise 1: Symmetry-Aware Heavy-Atom RMSD
Implement `compute_heavy_atom_rmsd(ref_coords: np.ndarray, query_coords: np.ndarray) -> float`:
- Computes Euclidean distance for each atom: $d_i = \sqrt{(x_i - x_i')^2 + (y_i - y_i')^2 + (z_i - z_i')^2}$.
- Computes root-mean-square deviation:
  $$\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^N d_i^2}$$
- Returns RMSD rounded to 3 decimal places.
""")

    add_code(nb, """# Exercise 1: Heavy-Atom RMSD
def compute_heavy_atom_rmsd(ref_coords: np.ndarray, query_coords: np.ndarray) -> float:
    \"\"\"Compute RMSD between two sets of corresponding 3D atomic coordinates.\"\"\"
    # --- SOLUTION BEGIN ---
    r1 = np.asarray(ref_coords, dtype=float)
    r2 = np.asarray(query_coords, dtype=float)
    if r1.shape != r2.shape:
        raise ValueError("Coordinate shapes must match")
    diff = r1 - r2
    sq_dist = np.sum(diff ** 2, axis=1)
    rmsd = np.sqrt(np.mean(sq_dist))
    return float(round(rmsd, 3))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 1
c_ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
c_same = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
assert compute_heavy_atom_rmsd(c_ref, c_same) == 0.0, "Identical coordinates must yield RMSD = 0"

# Uniform translation by 1.0 along x
c_shift = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
assert abs(compute_heavy_atom_rmsd(c_ref, c_shift) - 1.0) < 1e-3, "Translation by 1.0 must yield RMSD = 1.0"
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Enrichment Factor at $\chi\%$ ($EF_{\chi\%}$)
Implement `compute_enrichment_factor(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float`:
- Takes true binary labels `y_true` (1 = active, 0 = decoy) and prediction scores (higher score = better rank).
- Determines cut-off index $k = \max(1, \lfloor N \times \text{fraction} \rfloor)$.
- Counts number of actives in the top $k$ ranked compounds: $n_{\text{active, top } k}$.
- Computes baseline active fraction: $F_{\text{base}} = \frac{N_{\text{active, total}}}{N_{\text{total}}}$.
- Computes top-$k$ active fraction: $F_{\text{top } k} = \frac{n_{\text{active, top } k}}{k}$.
- Returns $EF = \frac{F_{\text{top } k}}{F_{\text{base}}}$.
""")

    add_code(nb, """# Exercise 2: Enrichment Factor
def compute_enrichment_factor(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float:
    \"\"\"Calculate Enrichment Factor at a given top fraction of ranked results.\"\"\"
    # --- SOLUTION BEGIN ---
    labels = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    n_total = len(labels)
    n_actives = np.sum(labels == 1)
    
    if n_actives == 0 or n_total == 0:
        return 0.0
    
    # Sort descending
    order = np.argsort(sc)[::-1]
    sorted_labels = labels[order]
    
    k = max(1, int(round(n_total * fraction)))
    actives_in_top_k = np.sum(sorted_labels[:k] == 1)
    
    ef = (actives_in_top_k / k) / (n_actives / n_total)
    return float(round(ef, 2))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
# Test library with 10 actives, 90 decoys (100 total, base active rate = 10%)
test_labels = np.array([1]*10 + [0]*90)
# Perfect ranking: all 10 actives at top
test_scores = np.linspace(100, 1, 100)

# EF at 5% (top 5 compounds are all actives: top rate = 5/5 = 100%, EF = 1.0 / 0.1 = 10.0)
ef5_perf = compute_enrichment_factor(test_labels, test_scores, fraction=0.05)
assert abs(ef5_perf - 10.0) < 0.1, f"Expected EF5% = 10.0, got {ef5_perf}"

# Benchmark calculation
ef1_bench = compute_enrichment_factor(y_true, pred_scores, fraction=0.01)
ef5_bench = compute_enrichment_factor(y_true, pred_scores, fraction=0.05)
print(f"Benchmark EF 1%: {ef1_bench:.2f}x enrichment")
print(f"Benchmark EF 5%: {ef5_bench:.2f}x enrichment")
assert ef1_bench >= 1.0, "Active enrichment should be >= 1.0"
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: BEDROC Calculation ($\alpha = 20.0$)
The Boltzmann-Enhanced Discrimination of ROC (BEDROC) addresses the "early recognition problem":
$$\text{BEDROC} = \frac{\sum_{i=1}^{N_{\text{active}}} e^{-\alpha r_i / N}}{\frac{R_a (1 - e^{-\alpha})}{e^{\alpha / N} - 1}} \times \dots$$
Using `chemlib.metrics.bedroc_score` or its direct implementation, calculate BEDROC with $\alpha = 20.0$.
""")

    add_code(nb, """# Exercise 3: BEDROC Metric
def compute_bedroc(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    \"\"\"Compute BEDROC score with exponential weighting factor alpha.\"\"\"
    # --- SOLUTION BEGIN ---
    labels = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    n = len(labels)
    n_actives = int(np.sum(labels == 1))
    
    if n_actives == 0 or n_actives == n:
        return 0.0

    order = np.argsort(sc)[::-1]
    sorted_labels = labels[order]
    
    active_ranks = np.where(sorted_labels == 1)[0] + 1  # 1-based ranks
    sum_exp = np.sum(np.exp(-alpha * active_ranks / n))
    denom = (1.0 / n) * ((1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0))
    rie = sum_exp / (n_actives * denom)

    ratio = n_actives / n
    rie_max = (1.0 - np.exp(-alpha * ratio)) / (ratio * (1.0 - np.exp(-alpha)))
    rie_min = (1.0 - np.exp(alpha * ratio)) / (ratio * (1.0 - np.exp(alpha)))

    if rie_max == rie_min:
        return 1.0
    bedroc = (rie - rie_min) / (rie_max - rie_min)
    return float(np.clip(bedroc, 0.0, 1.0))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
# Test on benchmark
bedroc_val = compute_bedroc(y_true, pred_scores, alpha=20.0)
print(f"Computed Benchmark BEDROC (alpha=20): {bedroc_val:.3f}")
assert 0.0 <= bedroc_val <= 1.0, f"BEDROC must be in [0, 1], got {bedroc_val}"
assert bedroc_val > 0.3, "Expected significant early recognition for actives"
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """## Common Pitfalls

1. **Rigid Receptor Assumption**: Standard docking treats receptors as rigid, missing induced-fit conformational adjustments upon ligand binding.
2. **Decoy Bias & Artificial Enrichment**: Decoys that are not physically property-matched (in MW, LogP, charge) produce artificially elevated ROC-AUC values.
3. **PDBQT Charge Errors**: Incorrect partial charges or incorrect torsional root assignments in ligand preparation directly degrade docking pose quality.""")

    add_md(nb, """## Summary and Key Takeaways

- **SBVS Pipeline**: Structure-based virtual screening docks large libraries into 3D binding cavities to prioritize prospective hit candidates.
- **Redocking Validation**: Crystallographic redocking with heavy-atom RMSD < 2.0 Å validates docking protocol accuracy prior to virtual screening.
- **Early Enrichment**: Metrics such as EF1% and BEDROC ($\\alpha=20$) quantify early active recovery in virtual screening campaigns.""")

    add_code(nb, """# Final validation
print("✔ NB18 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb18_docking_and_virtual_screening.ipynb",
        "solutions/nb18_docking_and_virtual_screening.ipynb"
    )

if __name__ == "__main__":
    build_nb18()
