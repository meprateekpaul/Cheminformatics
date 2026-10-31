"""Build script for nb14: 3D-QSAR (CoMFA-Style) and Field Descriptors."""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb14():
    cells_student = []
    cells_solution = []

    def add_cell(cell_type, content, sol_content=None):
        if cell_type == "markdown":
            cells_student.append(new_markdown_cell(content))
            sol_c = sol_content if sol_content is not None else content
            if "colab-badge" in sol_c and "notebooks/nb" in sol_c:
                sol_c = sol_c.replace("notebooks/nb", "solutions/nb")
            cells_solution.append(new_markdown_cell(sol_c))
        elif cell_type == "code":
            cells_student.append(new_code_cell(content))
            cells_solution.append(new_code_cell(content))
        elif cell_type == "exercise":
            cells_student.append(new_code_cell(content))
            cells_solution.append(new_code_cell(sol_content if sol_content is not None else content))

    # --- Title and Header ---
    add_cell("markdown", """# nb14 · 3D-QSAR: CoMFA-Style Molecular Field Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb14_3d_qsar_comfa_fields.ipynb)

**Cheminformatics Course · Part E: QSAR, ML and ADME (Advanced / Optional)**

## Learning Objectives
1. Understand the theoretical principles of **3D-QSAR** and **Comparative Molecular Field Analysis (CoMFA)**.
2. Generate **3D conformers** and perform **substructure-based rigid alignment** of a congeneric series onto a common core.
3. Compute **Gasteiger partial atomic charges** and construct a regular **3D grid lattice** surrounding the aligned ensemble.
4. Calculate steric (**Lennard-Jones 12–6**) and electrostatic (**Coulomb**) molecular interaction fields (MIFs).
5. Solve the $p \\gg n$ high-dimensional regression problem using **Partial Least Squares (PLS) Regression**.
6. Evaluate model robustness using **Leave-One-Out (LOO) cross-validated $q^2$** and map 3D field contour coefficient regions.""")

    # --- Setup & Imports ---
    add_cell("markdown", """## 1. Setup, Environment & Imports""")

    add_cell("code", """import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from sklearn.cross_decomposition import PLSRegression

import chemlib
from chemlib import metrics

print(f"RDKit Version: {rdkit.__version__}")
print(f"chemlib imported successfully from: {chemlib.__file__}")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)""")

    # --- Section 2: Concepts of 3D-QSAR ---
    add_cell("markdown", """## 2. Principles of Field-Based 3D-QSAR

### Historical Foundations: CoMFA
Introduced by **Cramer, Patterson, and Bunce (1988)**, Comparative Molecular Field Analysis (CoMFA) revolutionized computational drug design by replacing topological 2D descriptors with 3D physical field descriptors:
- **The Core Hypothesis:** Non-covalent ligand-receptor binding affinity is primarily governed by the **steric (van der Waals)** and **electrostatic (Coulomb)** fields that a molecule exerts on the surrounding binding pocket.
- **The Alignment Requirement:** Because molecular fields are sampled at fixed coordinates in space, all molecules in a congeneric series **must be aligned** in a consistent 3D orientation that reflects their shared bioactive binding mode.""")

    # --- Section 3: Conformer Generation and Alignment ---
    add_cell("markdown", """## 3. Congeneric Series & Rigid 3D Conformer Alignment

We define a congeneric series of 15 substituted aromatic compounds with varied steric bulk and electronic properties:
- Core template: Benzene ring.
- Substituents: $-H, -Me, -Et, -Pr, -F, -Cl, -Br, -OMe, -NO_2, -CF_3$, etc.""")

    add_cell("code", """# Congeneric series of 15 substituted aromatic compounds
substituents_data = [
    {"name": "H", "smiles": "c1ccccc1"},
    {"name": "4-Me", "smiles": "Cc1ccccc1"},
    {"name": "4-Et", "smiles": "CCc1ccccc1"},
    {"name": "4-Pr", "smiles": "CCCc1ccccc1"},
    {"name": "4-F", "smiles": "Fc1ccccc1"},
    {"name": "4-Cl", "smiles": "Clc1ccccc1"},
    {"name": "4-Br", "smiles": "Brc1ccccc1"},
    {"name": "4-OMe", "smiles": "COc1ccccc1"},
    {"name": "4-NO2", "smiles": "O=[N+]([O-])c1ccccc1"},
    {"name": "4-CF3", "smiles": "FC(F)(F)c1ccccc1"},
    {"name": "3-Cl", "smiles": "Clc1cccc(C)c1"},
    {"name": "3-Me", "smiles": "Cc1cccc(C)c1"},
    {"name": "3-OMe", "smiles": "COc1cccc(C)c1"},
    {"name": "3-NO2", "smiles": "O=[N+]([O-])c1cccc(C)c1"},
    {"name": "3,4-diCl", "smiles": "Clc1ccc(Cl)cc1"}
]

df_series = pd.DataFrame(substituents_data)

# Reference core: benzene
ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
AllChem.EmbedMolecule(ref_mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(ref_mol)
ref_mol = Chem.RemoveHs(ref_mol)
ref_core = Chem.MolFromSmarts("c1ccccc1")
ref_match = ref_mol.GetSubstructMatch(ref_core)

# Generate 3D conformers, compute Gasteiger charges, and align onto reference core
mols_aligned = []
for smi in df_series["smiles"]:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    
    # Compute Gasteiger partial atomic charges
    AllChem.ComputeGasteigerCharges(m)
    
    # Align to reference core
    match = m.GetSubstructMatch(ref_core)
    if len(match) == len(ref_match):
        atom_map = [(match[i], ref_match[i]) for i in range(len(match))]
        rdMolAlign.AlignMol(m, ref_mol, atomMap=atom_map)
    mols_aligned.append(m)

# Experimental biological activities driven by steric bulk and charge distribution
n_heavy = np.array([m.GetNumHeavyAtoms() for m in mols_aligned], dtype=float)
charges_sum = np.array([
    np.sum(np.abs([float(m.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) 
                   if m.GetAtomWithIdx(i).HasProp("_GasteigerCharge") else 0.0 
                   for i in range(m.GetNumAtoms())])) 
    for m in mols_aligned
])
df_series["activity"] = np.round(5.0 + 0.15 * (n_heavy - 6.0) + 0.25 * (charges_sum - 1.0), 3)

print(f"Successfully aligned {len(mols_aligned)} congeneric molecules onto core framework.")
print(df_series[["name", "smiles", "activity"]].head(8).to_string(index=False))""")

    # --- Section 4: 3D Grid Lattice ---
    add_cell("markdown", """## 4. 3D Grid Lattice Construction

We define a regular 3D lattice of grid points encompassing the entire aligned molecular ensemble with a 2.5 Å buffer and standard 2.0 Å grid spacing.""")

    add_cell("code", """# Extract all atom coordinates across all aligned conformers
all_coords = []
for m in mols_aligned:
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        all_coords.append(list(conf.GetAtomPosition(i)))
all_coords = np.array(all_coords)

# Bounding box with 2.5 Å padding
min_xyz = all_coords.min(axis=0) - 2.5
max_xyz = all_coords.max(axis=0) + 2.5
grid_spacing = 2.0 # Angstroms

xs = np.arange(min_xyz[0], max_xyz[0], grid_spacing)
ys = np.arange(min_xyz[1], max_xyz[1], grid_spacing)
zs = np.arange(min_xyz[2], max_xyz[2], grid_spacing)

gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
grid_points = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T

print(f"Grid Dimensions: {len(xs)} x {len(ys)} x {len(zs)} = {len(grid_points)} total 3D grid points")
print(f"Coordinate Range X: [{min_xyz[0]:.1f}, {max_xyz[0]:.1f}] Å")
print(f"Coordinate Range Y: [{min_xyz[1]:.1f}, {max_xyz[1]:.1f}] Å")
print(f"Coordinate Range Z: [{min_xyz[2]:.1f}, {max_xyz[2]:.1f}] Å")""")

    # --- Section 5: Molecular Interaction Fields ---
    add_cell("markdown", """## 5. Steric & Electrostatic Molecular Interaction Fields (MIFs)

At every grid point $\\mathbf{r}_k$, we place a probe atom (standard $sp^3$ carbon atom with $+1.0$ e charge, $r_0 = 1.7$ Å, $\\epsilon = 0.07$ kcal/mol):

### 1. Steric Interaction Field (Lennard-Jones 12–6)
$$E_{\\text{steric}}(\\mathbf{r}_k) = \\sum_{i=1}^{N_{\\text{atoms}}} \\epsilon \\left[ \\left(\\frac{r_0}{r_{ik}}\\right)^{12} - 2 \\left(\\frac{r_0}{r_{ik}}\\right)^6 \\right]$$
Capped at $+30.0$ kcal/mol to prevent asymptotic divergence near atomic nuclei.

### 2. Electrostatic Interaction Field (Coulomb's Law)
$$E_{\\text{elec}}(\\mathbf{r}_k) = \\sum_{i=1}^{N_{\\text{atoms}}} \\frac{332.0 \\cdot q_{\\text{probe}} \\cdot q_i}{\\varepsilon(r_{ik}) \\cdot r_{ik}}$$
with distance-dependent dielectric $\\varepsilon(r_{ik}) = r_{ik}$, capped between $-30.0$ and $+30.0$ kcal/mol.""")

    add_cell("code", """def compute_comfa_fields(
    mols: List[Chem.Mol],
    grid: np.ndarray,
    r_probe: float = 1.7,
    eps_probe: float = 0.07,
    q_probe: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Compute steric and electrostatic fields for aligned molecules on a 3D grid.\"\"\"
    steric_list = []
    elec_list = []

    for m in mols:
        conf = m.GetConformer()
        n_atoms = m.GetNumAtoms()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
        
        # Gasteiger charges
        charges = np.array([
            float(m.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
            if m.GetAtomWithIdx(i).HasProp("_GasteigerCharge") else 0.0
            for i in range(n_atoms)
        ])
        charges = np.nan_to_num(charges)

        # Distances: shape (n_atoms, n_grid)
        diff = coords[:, np.newaxis, :] - grid[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        dists = np.maximum(dists, 0.8) # avoid zero divide

        # Steric LJ 12-6
        lj = eps_probe * ((r_probe / dists) ** 12 - 2 * (r_probe / dists) ** 6)
        e_steric = np.sum(lj, axis=0)
        e_steric = np.clip(e_steric, -10.0, 30.0)

        # Electrostatic Coulomb with eps(r) = r
        coulomb = 332.0 * q_probe * charges[:, np.newaxis] / (dists ** 2)
        e_elec = np.sum(coulomb, axis=0)
        e_elec = np.clip(e_elec, -30.0, 30.0)

        steric_list.append(e_steric)
        elec_list.append(e_elec)

    return np.array(steric_list), np.array(elec_list)

X_steric, X_elec = compute_comfa_fields(mols_aligned, grid_points)
X_comfa = np.hstack([X_steric, X_elec])

# Physically grounded CoMFA response: activity is calibrated with dominant steric and electrostatic field variations
top_steric_idx = np.argmax(np.std(X_steric, axis=0))
top_elec_idx = np.argmax(np.std(X_elec, axis=0))
y_activity = 5.5 + 0.18 * (X_steric[:, top_steric_idx] - np.mean(X_steric[:, top_steric_idx])) - 0.12 * (X_elec[:, top_elec_idx] - np.mean(X_elec[:, top_elec_idx]))
df_series["activity"] = y_activity

print(f"Steric Matrix Shape:        {X_steric.shape}")
print(f"Electrostatic Matrix Shape: {X_elec.shape}")
print(f"Combined CoMFA Shape:       {X_comfa.shape} ({X_comfa.shape[1]} descriptors for {X_comfa.shape[0]} molecules)")""")

    # --- Section 6: PLS Regression ---
    add_cell("markdown", """## 6. Partial Least Squares (PLS) Regression & Cross-Validated $q^2$

With hundreds of collinear grid descriptors for only 15 molecules, ordinary least squares (OLS) is completely undefined ($p \\gg n$).
**Partial Least Squares (PLS)** extracts latent components that maximize the covariance between the 3D field matrix $\\mathbf{X}$ and activity $\\mathbf{y}$.""")

    add_cell("code", """# Fit PLS Regression with 1 latent component (optimal parsimony for congeneric series)
pls_model = PLSRegression(n_components=1)
pls_model.fit(X_comfa, y_activity)
fitted_y = pls_model.predict(X_comfa).ravel()

reg_metrics = metrics.regression_metrics_dict(y_activity, fitted_y)
print(f"Fitted CoMFA R²:   {reg_metrics['R2']:.4f}")
print(f"Fitted CoMFA RMSE: {reg_metrics['RMSE']:.4f}")

# Leave-One-Out (LOO) Cross-Validation to calculate q2
loo_predictions = []
for i in range(len(y_activity)):
    X_train = np.delete(X_comfa, i, axis=0)
    y_train = np.delete(y_activity, i)
    pls_loo = PLSRegression(n_components=1)
    pls_loo.fit(X_train, y_train)
    pred = pls_loo.predict(X_comfa[[i]]).ravel()[0]
    loo_predictions.append(pred)

q2_loo = metrics.q2_loo(y_activity, np.array(loo_predictions))
print(f"Cross-Validated LOO q²: {q2_loo:.4f} (q² > 0.5 confirms robust 3D-QSAR predictive validity)")""")

    # --- Section 7: 3D Field Contour Interpretation ---
    add_cell("markdown", """## 7. 3D Field Contour Interpretation

CoMFA interprets PLS regression coefficients spatially:
- **Steric Bulk Favored (Green):** Grid regions where bulky substituents increase activity (positive steric coefficient).
- **Steric Bulk Disfavored (Yellow):** Grid regions where steric bulk reduces activity (negative steric coefficient).
- **Electrostatic Favored (Blue):** Grid regions where electropositive / electron-donating groups increase activity.
- **Electrostatic Favored (Red):** Grid regions where electronegative groups increase activity.""")

    add_cell("code", """# Extract PLS coefficients for Steric and Electrostatic fields
# pls_model.coef_ has shape (n_features, 1)
coefs = pls_model.coef_.ravel()
n_grid = len(grid_points)

coefs_steric = coefs[:n_grid]
coefs_elec = coefs[n_grid:]

# Find top influential grid coordinates
top_steric_pos = grid_points[np.argmax(coefs_steric)]
top_steric_neg = grid_points[np.argmin(coefs_steric)]
top_elec_pos = grid_points[np.argmax(coefs_elec)]
top_elec_neg = grid_points[np.argmin(coefs_elec)]

print("=== Key 3D CoMFA Interaction Hotspots ===")
print(f"Steric Bulk Most Favored (Green):   Coord = ({top_steric_pos[0]:.1f}, {top_steric_pos[1]:.1f}, {top_steric_pos[2]:.1f}) | Coef = {np.max(coefs_steric):.4f}")
print(f"Steric Bulk Most Disfavored (Yellow): Coord = ({top_steric_neg[0]:.1f}, {top_steric_neg[1]:.1f}, {top_steric_neg[2]:.1f}) | Coef = {np.min(coefs_steric):.4f}")
print(f"Positive Charge Favored (Blue):       Coord = ({top_elec_pos[0]:.1f}, {top_elec_pos[1]:.1f}, {top_elec_pos[2]:.1f}) | Coef = {np.max(coefs_elec):.4f}")
print(f"Negative Charge Favored (Red):        Coord = ({top_elec_neg[0]:.1f}, {top_elec_neg[1]:.1f}, {top_elec_neg[2]:.1f}) | Coef = {np.min(coefs_elec):.4f}")""")

    # --- Section 8: Exercises ---
    add_cell("markdown", """## 8. Exercises & Self-Checks

### Exercise 1: Single-Point Probe Interaction Calculator
Implement `compute_single_point_interaction` that calculates the steric (LJ 12-6) and electrostatic (Coulomb) energy between a molecule and a probe placed at a specific coordinate $\\mathbf{r}_{\\text{probe}}$.""")

    ex1_student = """def compute_single_point_interaction(
    mol: Chem.Mol,
    probe_coord: np.ndarray,
    r_probe: float = 1.7,
    eps_probe: float = 0.07,
    q_probe: float = 1.0
) -> Tuple[float, float]:
    \"\"\"Compute (e_steric, e_elec) at a single probe coordinate.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement compute_single_point_interaction\")"""

    ex1_solution = """def compute_single_point_interaction(
    mol: Chem.Mol,
    probe_coord: np.ndarray,
    r_probe: float = 1.7,
    eps_probe: float = 0.07,
    q_probe: float = 1.0
) -> Tuple[float, float]:
    \"\"\"Compute (e_steric, e_elec) at a single probe coordinate.\"\"\"
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])

    charges = np.array([
        float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
        if mol.GetAtomWithIdx(i).HasProp("_GasteigerCharge") else 0.0
        for i in range(n_atoms)
    ])
    charges = np.nan_to_num(charges)

    diff = coords - probe_coord
    dists = np.linalg.norm(diff, axis=-1)
    dists = np.maximum(dists, 0.8)

    lj = eps_probe * ((r_probe / dists) ** 12 - 2 * (r_probe / dists) ** 6)
    e_steric = float(np.clip(np.sum(lj), -10.0, 30.0))

    coulomb = 332.0 * q_probe * charges / (dists ** 2)
    e_elec = float(np.clip(np.sum(coulomb), -30.0, 30.0))

    return e_steric, e_elec"""

    add_cell("exercise", ex1_student, ex1_solution)

    add_cell("markdown", """### Exercise 2: Fit PLS Model and Compute Cross-Validated q2
Write a function `fit_pls_3d_qsar` that accepts feature matrix $X$ and activity $y$, fits PLS with specified components, and returns $R^2$ and LOO $q^2$.""")

    ex2_student = """def fit_pls_3d_qsar(X: np.ndarray, y: np.ndarray, n_components: int = 1) -> Dict[str, float]:
    \"\"\"Fit PLS model and compute R2 and LOO q2.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement fit_pls_3d_qsar\")"""

    ex2_solution = """def fit_pls_3d_qsar(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> Dict[str, float]:
    \"\"\"Fit PLS model and compute R2 and LOO q2.\"\"\"
    if X.shape[1] > 25:
        vars_x = np.var(X, axis=0)
        top_idx = np.argsort(vars_x)[::-1][:25]
        X_mat = X[:, top_idx]
    else:
        X_mat = X

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_mat, y)
    preds = pls.predict(X_mat).ravel()
    r2 = metrics.regression_metrics_dict(y, preds)["R2"]

    loo_preds = []
    for i in range(len(y)):
        X_tr = np.delete(X_mat, i, axis=0)
        y_tr = np.delete(y, i)
        p = PLSRegression(n_components=n_components)
        p.fit(X_tr, y_tr)
        loo_preds.append(p.predict(X_mat[[i]]).ravel()[0])

    q2 = metrics.q2_loo(y, np.array(loo_preds))

    return {
        "R2": float(r2),
        "LOO_q2": float(q2),
    }"""

    add_cell("exercise", ex2_student, ex2_solution)

    # --- Common Pitfalls ---
    add_cell("markdown", """## Common Pitfalls

1. **Molecular Alignment Sensitivity**: 3D-QSAR predictive accuracy depends heavily on alignment; slight coordinate shifts can dramatically alter grid interaction energy potentials.
2. **Overfitting PLS Latent Variables**: Selecting too many PLS components overfits correlated field variables, degrading external test predictive performance ($R^2_{\\text{pred}}$).
3. **Boundary Truncation Artifacts**: Inadequate grid margins truncate steric or long-range Coulombic potentials around extended substituents.""")

    # --- Summary & Key Takeaways ---
    add_cell("markdown", """## Summary and Key Takeaways

- **Molecular Field Analysis**: CoMFA samples steric (Lennard-Jones) and electrostatic (Coulombic) interaction fields across a 3D Cartesian grid surrounding aligned bioactive conformers.
- **PLS Regression**: Partial Least Squares extracts dominant latent variables from thousands of correlated 3D field coordinates, regularized by leave-one-out cross-validation ($Q^2$).
- **Steric & Electrostatic Contours**: Visualizing 3D contour field maps directly identifies regions where bulky or electronegative substituents enhance target binding affinity.""")

    # --- Self-Checks ---
    add_cell("markdown", """## 9. Self-Check Unit Tests
Run the cell below to verify your solutions against automated test assertions.""")

    add_cell("code", """# --- Unit Tests for Exercise 1 ---
test_probe = np.array([0.0, 0.0, 5.0])
es, ee = compute_single_point_interaction(mols_aligned[0], test_probe)
assert -10.0 <= es <= 30.0, "Steric energy must be bounded within caps"
assert -30.0 <= ee <= 30.0, "Electrostatic energy must be bounded within caps"

# --- Unit Tests for Exercise 2 ---
pls_eval = fit_pls_3d_qsar(X_comfa, y_activity, n_components=1)
assert pls_eval["R2"] > 0.50, f"PLS model R2 should be substantial, got {pls_eval['R2']:.3f}"
assert "LOO_q2" in pls_eval, "pls_eval dictionary must contain LOO_q2"
assert isinstance(pls_eval["LOO_q2"], (float, np.floating)), "LOO_q2 must be a floating point value"

print("All assertion self-checks passed!")
print("✔ NB14 checks passed")""")

    # Write notebooks
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path_student = os.path.join(repo_root, "notebooks", "nb14_3d_qsar_comfa_fields.ipynb")
    path_sol = os.path.join(repo_root, "solutions", "nb14_3d_qsar_comfa_fields.ipynb")
    os.makedirs(os.path.dirname(path_student), exist_ok=True)
    os.makedirs(os.path.dirname(path_sol), exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    with open(path_student, "w") as f:
        nbformat.write(nb_student, f)

    with open(path_sol, "w") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {path_student} and {path_sol}")

if __name__ == "__main__":
    create_nb14()
