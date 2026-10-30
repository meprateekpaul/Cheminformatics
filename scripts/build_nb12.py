"""Build script for nb12: QSAR Foundations - From Crum-Brown to Hansch & Free-Wilson."""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb12():
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
    add_cell("markdown", """# nb12 · QSAR Foundations: From Crum-Brown to Hansch
**Cheminformatics Course · Part E: QSAR, ML and ADME**

### Objectives
1. Explain the **iterative molecular optimization loop** and the philosophical rationale for QSAR (cost savings, 3Rs principle, green chemistry).
2. Contrast qualitative **Structure-Activity Relationships (SAR)** with quantitative **QSAR**, recalling the Crum-Brown & Fraser (1868) foundation $\\Phi = f(C)$.
3. Implement classical **Hansch analysis** comparing linear and **parabolic models**, and derive the optimal lipophilicity $\\log P^*$.
4. Implement **Free-Wilson analysis** on congeneric series, handling indicator variables, reference compound choice, and rank deficiency (incorporating errata note E4).
5. Work through **$\\log P$ additivity** using substituent $\\pi$-constants and analyze cases where additivity breaks down (ortho effects, intramolecular H-bonding, cyclophanes).
6. Validate multiple linear regression (MLR) models according to **OECD principles**: $R^2$, adjusted $R^2$, standard error $s$, $F$-test, **Leave-One-Out (LOO) $q^2$**, **$Y$-randomization**, and the **Williams plot** (applicability domain).""")

    # --- Setup & Imports ---
    add_cell("markdown", """## 1. Setup, Environment & Imports""")

    add_cell("code", """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

import chemlib
from chemlib import metrics

print(f"Statsmodels Version: {sm.__version__}")
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

    # --- Section 2: Foundations & Workflow ---
    add_cell("markdown", """## 2. Theoretical Foundations: Crum-Brown to Hansch

### The Molecular Discovery Cycle
The standard drug discovery pipeline is an iterative cycle:
$$\\text{Design} \\longrightarrow \\text{Synthesize} \\longrightarrow \\text{Test} \\longrightarrow \\text{Analyze} \\longrightarrow \\text{Model} \\longrightarrow \\text{Design}$$

### Why QSAR?
1. **Resource Conservation:** Reduces the synthesis of inactive analogs, saving reagents and chemist-hours.
2. **The 3Rs Principle:** **Replace, Reduce, and Refine** animal experimentation in pharmacological testing.
3. **Green Chemistry & REACH:** Early in-silico hazard and toxicity assessment before scaling up chemical production.

### Historical Evolution
- **Crum-Brown and Fraser (1868):** First proposed that biological activity $\\Phi$ is a function of chemical constitution $C$:
  $$\\Phi = f(C), \\quad \\Delta \\Phi \\longleftrightarrow \\Delta C$$
- **Meyer & Overton (1899):** Discovered that the narcotic potency of organic molecules correlates directly with their olive oil/water partition coefficient.
- **Corwin Hansch (1962):** Integrated physical organic parameters (hydrophobic $\\pi$, electronic Hammett $\\sigma$, and steric Taft $E_s$) into multiparameter regression models.""")

    # --- Section 3: Hansch Analysis ---
    add_cell("markdown", """## 3. Hansch Analysis: Linear vs Parabolic Models

Hansch proposed that biological activity $\\log(1/C)$ (where $C$ is molar concentration producing a standard response, e.g. $IC_{50}$) depends on hydrophobic and electronic properties:

### 1. Linear Hansch Model
$$\\log(1/C) = k_1 \\log P + k_2 \\sigma + k_3$$
Valid only over narrow ranges of lipophilicity where partitioning is the sole rate-limiting step.

### 2. Parabolic Hansch Model
$$\\log(1/C) = -k_1 (\\log P - \\log P^*)^2 + k_2 \\sigma + k_3 = a (\\log P)^2 + b \\log P + c \\sigma + d$$
where:
$$a = -k_1, \\quad b = 2 k_1 \\log P^* \\implies \\log P^* = -\\frac{b}{2a}$$

### Why is the relationship parabolic?
- **Hydrophilic extremes (low $\\log P$):** Molecules cannot cross hydrophobic lipid bilayer membranes and remain trapped in extracellular aqueous compartments.
- **Lipophilic extremes (high $\\log P$):** Molecules become tightly sequestered inside membrane interiors or precipitate out of solution, failing to reach intracellular targets.
- **Optimum $\\log P^*$:** Maximizes the probability that a molecule traverses biological barriers to reach the receptor.""")

    add_cell("code", """# Load Hansch dataset
data_hansch = find_data_file("d13_hansch_series.csv")
df_hansch = pd.read_csv(data_hansch)
print("=== D13 Hansch Substituent Series ===")
print(df_hansch.head(10).to_string(index=False))

# 1. Fit Linear Hansch Model: log(1/C) = k1*logP + k2*sigma + const
X_lin = sm.add_constant(df_hansch[["logP", "hammett_sigma"]])
y_h = df_hansch["log_1_C_obs"]
model_lin = sm.OLS(y_h, X_lin).fit()

# 2. Fit Parabolic Hansch Model: log(1/C) = a*(logP^2) + b*logP + c*sigma + d
df_hansch["logP_sq"] = df_hansch["logP"] ** 2
X_parab = sm.add_constant(df_hansch[["logP", "logP_sq", "hammett_sigma"]])
model_parab = sm.OLS(y_h, X_parab).fit()

# Calculate optimal logP* = -b / (2*a)
b_lp = model_parab.params["logP"]
a_lp2 = model_parab.params["logP_sq"]
opt_logP = -b_lp / (2.0 * a_lp2)

print(f"Linear Model:    R² = {model_lin.rsquared:.3f} | AIC = {model_lin.aic:.2f}")
print(f"Parabolic Model: R² = {model_parab.rsquared:.3f} | AIC = {model_parab.aic:.2f}")
print(f"Fitted Optimum logP*: {opt_logP:.2f} (Quadratic coefficient a = {a_lp2:.3f}, p = {model_parab.pvalues['logP_sq']:.4f})")""")

    # --- Section 4: Free-Wilson Analysis ---
    add_cell("markdown", """## 4. Free–Wilson Analysis on Congeneric Series

### The Free-Wilson Additive Principle
Free and Wilson (1964) modeled biological activity as the purely additive sum of substituent contributions at specific substitution positions:
$$\\log(1/C) = \\mu + \\sum_{i} \\sum_{j} a_{ij} X_{ij}$$
where $\\mu$ is the baseline activity of the unsubstituted parent molecule, and $a_{ij}$ is the activity contribution of substituent $j$ at position $i$.

### Reference Compound & Rank Deficiency
For $m$ substituents at position 1 and $n$ substituents at position 2, the indicator matrix contains $1 + m + n$ columns. However, because substituents at each position are mutually exclusive (sum of indicators = 1), this matrix is **rank deficient** (multicollinear).
To resolve this, we designate reference substituents (e.g. $R_1=H, R_2=H$) whose contributions are set to zero, leaving $(m - 1) + (n - 1)$ independent dummy columns.""")

    add_cell("code", """# Load Free-Wilson dataset
data_fw = find_data_file("d14_free_wilson_series.csv")
df_fw = pd.read_csv(data_fw)
print("=== D14 Free-Wilson Congeneric Series ===")
print(df_fw.head(8).to_string(index=False))

# Construct indicator design matrix with R1=H and R2=H as reference baseline
r1_dummies = pd.get_dummies(df_fw["R1"], prefix="R1", drop_first=True, dtype=float)
r2_dummies = pd.get_dummies(df_fw["R2"], prefix="R2", drop_first=True, dtype=float)
X_fw = sm.add_constant(pd.concat([r1_dummies, r2_dummies], axis=1))
y_fw = df_fw["pActivity_obs"]

model_fw = sm.OLS(y_fw, X_fw).fit()

print(f"Free-Wilson Model R²: {model_fw.rsquared:.4f} | Adjusted R²: {model_fw.rsquared_adj:.4f}")
print(f"Baseline Activity (R1=H, R2=H): mu = {model_fw.params['const']:.3f}")
print("Substituent Contributions:")
for col in X_fw.columns:
    if col != "const":
        print(f"  {col:<12}: {model_fw.params[col]:+.3f} (p = {model_fw.pvalues[col]:.2e})")""")

    # --- Section 5: logP Additivity Worked Example ---
    add_cell("markdown", """## 5. $\\log P$ Additivity: Worked Example from Lecture 06

The hydrophobic substituent constant $\\pi_X$ is defined as:
$$\\pi_X = \\log P_{RX} - \\log P_{RH}$$

### Calculating $\\log P$ of $m$-Chlorotoluene via 3 Independent Routes
- **Route 1 (from Benzene):**
  $$\\log P = \\log P_{\\text{benzene}} + \\pi_{\\text{Me}} + \\pi_{\\text{Cl}} = 2.13 + 0.71 + 0.56 = 3.40$$
- **Route 2 (from Toluene):**
  $$\\log P = \\log P_{\\text{toluene}} + \\pi_{\\text{Cl}} = 2.69 + 0.76 = 3.45$$
- **Route 3 (from Chlorobenzene):**
  $$\\log P = \\log P_{\\text{chlorobenzene}} + \\pi_{\\text{Me}} = 2.84 + 0.51 = 3.35$$
- **Experimental Value:** $3.28$
- **Spread:** $3.35$ to $3.45$ (Notice that empirical $\\pi$ constants depend subtly on the reference aromatic ring system).""")

    add_cell("code", """# Verify logP additivity values from lecture slides
r1_benzene = 2.13 + 0.71 + 0.56
r2_toluene = 2.69 + 0.76
r3_chlorobenzene = 2.84 + 0.51
exp_val = 3.28

# Compute RDKit Crippen MolLogP for m-chlorotoluene
mol_mct = Chem.MolFromSmiles("Cc1cccc(Cl)c1")
rdkit_logp = Crippen.MolLogP(mol_mct)

print(f"Route 1 (Benzene):       {r1_benzene:.2f}")
print(f"Route 2 (Toluene):       {r2_toluene:.2f}")
print(f"Route 3 (Chlorobenzene): {r3_chlorobenzene:.2f}")
print(f"Experimental Value:      {exp_val:.2f}")
print(f"RDKit Wildman-Crippen:   {rdkit_logp:.2f}")

assert abs(r1_benzene - 3.40) < 1e-4
assert abs(r2_toluene - 3.45) < 1e-4
assert abs(r3_chlorobenzene - 3.35) < 1e-4""")

    # --- Section 6: Non-Additivity ---
    add_cell("markdown", """## 6. When Additivity Fails: Non-Additive Structural Effects

Substituent additivity fails systematically in several important chemical regimes:
1. **Ortho Effects:** Steric hindrance causes substituents to rotate out of the plane of the aromatic ring, quenching resonance conjugation.
2. **Intramolecular Hydrogen Bonding:** E.g., Salicylic acid vs $p$-hydroxybenzoic acid. Intramolecular H-bonds shield polar groups from water, dramatically increasing apparent lipophilicity.
3. **Push–Pull Electronic Resonance:** E.g., 4-nitroaniline exhibits strong direct resonance between amino and nitro groups.
4. **Cyclophanes ($p$-xylene vs $[2.2]$paracyclophane):**
   - **$p$-xylene:** Surface area $\\approx 150.3\\text{ Å}^2$, experimental $\\log P = 3.15$.
   - **$[2.2]$paracyclophane:** Surface area $\\approx 196.6\\text{ Å}^2$, experimental $\\log P = 2.33$.
   - Despite being significantly larger in surface area, $[2.2]$paracyclophane has a **lower** $\\log P$ due to electronic transannular interactions between stacked aromatic decks!""")

    add_cell("code", """# Case study: p-xylene vs [2.2]paracyclophane
cyclophane_data = [
    {"compound": "p-xylene", "smiles": "Cc1ccc(C)cc1", "surface_area": 150.3, "logP_exp": 3.15},
    {"compound": "[2.2]paracyclophane", "smiles": "C1Cc2ccc(cc2)CCc3ccc1cc3", "surface_area": 196.6, "logP_exp": 2.33}
]
df_cyclo = pd.DataFrame(cyclophane_data)
df_cyclo["rdkit_logp"] = df_cyclo["smiles"].apply(lambda s: Crippen.MolLogP(Chem.MolFromSmiles(s)))

print("=== Paracyclophane Non-Additivity Paradox ===")
print(df_cyclo[["compound", "surface_area", "logP_exp", "rdkit_logp"]].to_string(index=False))""")

    # --- Section 7: MLR Metrics & OECD Validation ---
    add_cell("markdown", """## 7. Model Validation & OECD Principles

The **OECD Principles for the Validation of (Q)SARs** require:
1. A defined endpoint.
2. An unambiguous algorithm.
3. A defined domain of applicability.
4. Appropriate measures of goodness-of-fit, robustness, and predictivity.
5. A mechanistic interpretation, if possible.

### Statistical Validation Tools
1. **Goodness of Fit:** $R^2$, Adjusted $R^2$, Standard Error of Estimate:
   $$s = \\sqrt{\\frac{\\sum (y - \\hat{y})^2}{N - p - 1}}$$
2. **Robustness (Internal Predictivity):** Leave-One-Out (LOO) cross-validation $q^2$:
   $$q^2 = 1 - \\frac{\\sum_{i=1}^N (y_i - \\hat{y}_{i(-i)})^2}{\\sum_{i=1}^N (y_i - \\bar{y})^2}$$
3. **Chance Correlation Test:** **$Y$-randomization**. Scramble target $y$ randomly 50–100 times and refit; mean $R^2_{\\text{rand}}$ should collapse to $<0.2$.
4. **Applicability Domain (Williams Plot):** Standardized residuals vs leverage $h_i = \\mathbf{x}_i (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{x}_i^T$ with warning leverage $h^* = \\frac{3(p + 1)}{N}$.""")

    add_cell("code", """# Compute Leave-One-Out (LOO) q2 for Free-Wilson model
y_vals = y_fw.values
preds_loo = []

for i in range(len(y_vals)):
    X_train = X_fw.drop(index=i)
    y_train = y_fw.drop(index=i)
    m = sm.OLS(y_train, X_train).fit()
    pred_val = float(m.predict(X_fw.iloc[[i]]).values[0])
    preds_loo.append(pred_val)

q2_val = metrics.q2_loo(y_vals, np.array(preds_loo))
print(f"Free-Wilson Leave-One-Out q²: {q2_val:.4f} (q² > 0.6 confirms high internal robustness)")

# Compute Leverage for Williams plot
X_mat = X_fw.values
hat_matrix = X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T
leverages = np.diag(hat_matrix)
p_num = X_fw.shape[1] - 1
n_samples = len(y_fw)
h_warning = 3.0 * (p_num + 1) / n_samples

print(f"Number of parameters p = {p_num}, Sample size N = {n_samples}")
print(f"Warning Leverage Threshold h* = 3(p+1)/N = {h_warning:.3f}")
print(f"Max observed leverage in dataset:        {np.max(leverages):.3f}")""")

    # --- Section 8: Exercises ---
    add_cell("markdown", """## 8. Exercises & Self-Checks

### Exercise 1: Parabolic Hansch Optimization
Write a function `fit_parabolic_hansch` that accepts a DataFrame with columns `['logP', 'hammett_sigma', 'log_1_C_obs']`, fits the parabolic OLS model, and returns a dictionary containing $R^2$, standard error $s$, and the calculated optimum $\\log P^*$.""")

    ex1_student = """def fit_parabolic_hansch(df: pd.DataFrame) -> Dict[str, float]:
    \"\"\"Fit parabolic Hansch model and return R2, standard error s, and optimal logP*.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement fit_parabolic_hansch\")"""

    ex1_solution = """def fit_parabolic_hansch(df: pd.DataFrame) -> Dict[str, float]:
    \"\"\"Fit parabolic Hansch model and return R2, standard error s, and optimal logP*.\"\"\"
    df_fit = df.copy()
    df_fit["logP_sq"] = df_fit["logP"] ** 2
    X = sm.add_constant(df_fit[["logP", "logP_sq", "hammett_sigma"]])
    y = df_fit["log_1_C_obs"]

    m = sm.OLS(y, X).fit()
    r2 = float(m.rsquared)
    b_lp = m.params["logP"]
    a_lp2 = m.params["logP_sq"]
    opt_lp = float(-b_lp / (2.0 * a_lp2))

    # Standard error s = sqrt(SSR / (N - p - 1))
    s_err = float(np.sqrt(m.ssr / m.df_resid))

    return {
        "R2": r2,
        "Standard_Error_s": s_err,
        "Optimal_logP": opt_lp,
    }"""

    add_cell("exercise", ex1_student, ex1_solution)

    add_cell("markdown", """### Exercise 2: Y-Randomization Validation
Implement `run_y_randomization` to perform $N$ iterations of response scrambling, refitting the model each time, and returning the mean and maximum randomized $R^2$ values.""")

    ex2_student = """def run_y_randomization(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 50,
    seed: int = 42
) -> Dict[str, float]:
    \"\"\"Run Y-randomization test and return mean and max randomized R2.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement run_y_randomization\")"""

    ex2_solution = """def run_y_randomization(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 50,
    seed: int = 42
) -> Dict[str, float]:
    \"\"\"Run Y-randomization test and return mean and max randomized R2.\"\"\"
    rng = np.random.RandomState(seed)
    r2_list = []
    y_arr = y.values.copy()

    for _ in range(n_iterations):
        y_perm = rng.permutation(y_arr)
        m = sm.OLS(y_perm, X).fit()
        r2_list.append(m.rsquared)

    return {
        "mean_R2_rand": float(np.mean(r2_list)),
        "max_R2_rand": float(np.max(r2_list)),
    }"""

    add_cell("exercise", ex2_student, ex2_solution)

    # --- Self-Checks ---
    add_cell("markdown", """## 9. Self-Check Unit Tests
Run the cell below to verify your solutions against automated test assertions.""")

    add_cell("code", """# --- Unit Tests for Exercise 1 ---
res_h = fit_parabolic_hansch(df_hansch)
assert 0.80 <= res_h["R2"] <= 0.95, f"Expected parabolic R2 ~ 0.85, got {res_h['R2']:.3f}"
assert 2.3 <= res_h["Optimal_logP"] <= 2.8, f"Expected optimal logP* ~ 2.5-2.6, got {res_h['Optimal_logP']:.2f}"
assert res_h["Standard_Error_s"] < 0.25, f"Standard error s should be small, got {res_h['Standard_Error_s']:.3f}"

# --- Unit Tests for Exercise 2 ---
y_rand_res = run_y_randomization(X_fw, y_fw, n_iterations=30, seed=123)
assert y_rand_res["mean_R2_rand"] < 0.50, f"Randomized R2 should collapse, got {y_rand_res['mean_R2_rand']:.3f}"
assert y_rand_res["max_R2_rand"] < model_fw.rsquared, "Randomized model should never beat real model"

print("All assertion self-checks passed!")
print("✔ NB12 checks passed")""")

    # Write notebooks
    os.makedirs("Cheminformatics/notebooks", exist_ok=True)
    os.makedirs("Cheminformatics/solutions", exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    path_student = "Cheminformatics/notebooks/nb12_qsar_foundations_hansch.ipynb"
    path_sol = "Cheminformatics/solutions/nb12_qsar_foundations_hansch.ipynb"

    with open(path_student, "w") as f:
        nbformat.write(nb_student, f)

    with open(path_sol, "w") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {path_student} and {path_sol}")

if __name__ == "__main__":
    create_nb12()
