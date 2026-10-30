import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb21():
    nb = create_notebook()

    add_md(nb, """# Notebook 21: Graph Neural Networks for Molecular Property Prediction

**Part G: Capstone and Advanced Topics**  
**Estimated Time:** ~120 min  
**Level:** Advanced / Optional (●●●)

---

## 🎯 Learning Objectives
1. **Molecules as Attributed Graphs:** Represent molecular structures as mathematical graphs $G = (V, E)$ with atom feature vectors $\mathbf{h}_v$ and bond connectivity matrices $\mathbf{A}$.
2. **Spectral & Spatial Graph Convolutions:** Understand the derivation and physical intuition behind the Kipf & Welling **Graph Convolutional Network (GCN)**:
   $$\mathbf{H}^{(l+1)} = \sigma \left( \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
3. **Message Passing Paradigm:** Implement neighborhood aggregation and message passing from scratch in pure NumPy / Python.
4. **Global Graph Readout (Pooling):** Aggregate atom-level embeddings into whole-molecule graph vectors via mean and sum pooling.
5. **Aqueous Solubility Prediction (Delaney ESOL):** Train a graph network to predict experimental aqueous solubility ($\log S$).
6. **Benchmark Comparison:** Compare GNN accuracy against classical **Random Forest + ECFP4** baselines on a scaffold-split test set.
7. **Explainability & Atomic Attribution:** Map atom-level learned representations to identify solubilizing (polar) versus lipophilic (aromatic) pharmacophoric centers.
""")

    add_md(nb, """## 1. Prerequisites & Environment Setup""")

    add_code(nb, """import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, metrics as skmetrics

# Include chemlib
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, AllChem
    import chemlib
    from chemlib import graphs, metrics
    print(f"RDKit {Chem.__version__} and chemlib successfully loaded.")
except ImportError:
    print("Running in analytical mode.")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 2. Molecular Graph Theory: Adjacency, Degree, and Kipf-Welling Normalization

A molecule with $N$ heavy atoms is represented as a graph $G = (V, E)$:
- **Adjacency Matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$:**
  $$A_{ij} = \begin{cases} 1 & \text{if atom } i \text{ is bonded to atom } j \\ 0 & \text{otherwise} \end{cases}$$
- **Self-Loops & Augmented Adjacency $\mathbf{\tilde{A}}$:**
  In message passing, each atom must retain information about its own current state. We add the identity matrix $\mathbf{I}_N$:
  $$\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$$
- **Augmented Degree Matrix $\mathbf{\tilde{D}}$:**
  Diagonal matrix where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij} = \text{degree}(i) + 1$.
- **Symmetric Normalization:**
  Multiplying by $\mathbf{\tilde{A}}$ alone would cause feature magnitudes to blow up for high-degree atoms (e.g. quaternary carbons). We normalize symmetrically:
  $$\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2}$$
  The $(i, j)$-th entry is simply:
  $$\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{D}_{ii} \tilde{D}_{jj}}}$$
""")

    add_code(nb, """# Demonstrate Kipf-Welling normalization on Aspirin
aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"
mol_asp = Chem.MolFromSmiles(aspirin_smi)
n_atoms = mol_asp.GetNumHeavyAtoms()

# Standard adjacency
A = Chem.GetAdjacencyMatrix(mol_asp)

# Augmented adjacency with self-loops
A_tilde = A + np.eye(n_atoms)

# Augmented degree vector
degrees = np.sum(A_tilde, axis=1)
deg_inv_sqrt = 1.0 / np.sqrt(degrees)
D_inv_sqrt = np.diag(deg_inv_sqrt)

# Symmetric normalized adjacency
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

print(f"Aspirin has {n_atoms} heavy atoms.")
print("Sample row of symmetrically normalized adjacency A_hat (Atom 0):")
print(np.round(A_hat[0, :], 3))
""")

    add_md(nb, """## 3. Atom-Level Featurization Pipeline

We represent each atom $v \in V$ as a fixed-length feature vector $\mathbf{h}_v \in \mathbb{R}^d$:
1. **Element Type (One-Hot, 8 categories):** Carbon, Nitrogen, Oxygen, Sulfur, Fluorine, Chlorine, Bromine, Other.
2. **Degree (One-Hot, 5 categories):** $0, 1, 2, 3, \ge 4$ bonded heavy neighbors.
3. **Formal Charge:** Integer charge ($0, +1, -1$).
4. **Hybridization (One-Hot, 4 categories):** $sp, sp^2, sp^3$, Other.
5. **Aromaticity:** Binary flag ($1$ if atom is part of an aromatic ring, $0$ otherwise).
6. **Hydrogen Count:** Number of implicit + explicit attached hydrogens.

This yields a rich $d = 20$-dimensional initial feature vector per atom: $\mathbf{H}^{(0)} \in \mathbb{R}^{N \times 20}$.
""")

    add_code(nb, """ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br"]
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]

def atom_to_feature_vector(atom: Chem.Atom) -> np.ndarray:
    \"\"\"Convert an RDKit Atom into a 20-dimensional feature vector.\"\"\"
    feats = []
    # 1. Element one-hot (8 dims)
    sym = atom.GetSymbol()
    elem_oh = [1.0 if sym == e else 0.0 for e in ELEMENTS]
    elem_oh.append(1.0 if sym not in ELEMENTS else 0.0)  # Other
    feats.extend(elem_oh)

    # 2. Degree one-hot (5 dims: 0, 1, 2, 3, >=4)
    deg = atom.GetTotalDegree()
    deg_oh = [1.0 if deg == d else 0.0 for d in range(4)]
    deg_oh.append(1.0 if deg >= 4 else 0.0)
    feats.extend(deg_oh)

    # 3. Formal charge (1 dim)
    feats.append(float(atom.GetFormalCharge()))

    # 4. Hybridization one-hot (4 dims)
    hyb = atom.GetHybridization()
    hyb_oh = [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS]
    hyb_oh.append(1.0 if hyb not in HYBRIDIZATIONS else 0.0)
    feats.extend(hyb_oh)

    # 5. Aromaticity (1 dim)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)

    # 6. Hydrogen count (1 dim)
    feats.append(float(atom.GetTotalNumHs()))

    return np.array(feats, dtype=np.float32)

def mol_to_graph_data(mol: Chem.Mol):
    \"\"\"Extract atom feature matrix H and normalized adjacency A_hat from a molecule.\"\"\"
    n = mol.GetNumHeavyAtoms()
    H = np.array([atom_to_feature_vector(mol.GetAtomWithIdx(i)) for i in range(n)], dtype=np.float32)
    A = Chem.GetAdjacencyMatrix(mol)
    A_tilde = A + np.eye(n, dtype=np.float32)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    A_hat = d_inv_sqrt @ A_tilde @ d_inv_sqrt
    return H, A_hat

H_asp, A_hat_asp = mol_to_graph_data(mol_asp)
print(f"Atom feature matrix shape: {H_asp.shape} (N={H_asp.shape[0]} atoms, d={H_asp.shape[1]} features)")
print(f"Normalized adjacency shape: {A_hat_asp.shape}")
""")

    add_md(nb, """## 4. Message Passing & Global Pooling Readout

### Forward Propagation across $L$ GCN Layers:
$$\mathbf{H}^{(1)} = \text{ReLU}\left( \mathbf{\hat{A}} \mathbf{H}^{(0)} \mathbf{W}^{(0)} + \mathbf{b}^{(0)} \right)$$
$$\mathbf{H}^{(2)} = \text{ReLU}\left( \mathbf{\hat{A}} \mathbf{H}^{(1)} \mathbf{W}^{(1)} + \mathbf{b}^{(1)} \right)$$

### Global Graph Pooling (Readout):
To predict a whole-molecule property (e.g. aqueous solubility $\log S$), we must aggregate all node embeddings into a single fixed-length molecular vector $\mathbf{h}_G$:
$$\mathbf{h}_G = \text{MeanPool}(\mathbf{H}^{(L)}) = \frac{1}{N} \sum_{v=1}^N \mathbf{h}_v^{(L)}$$

### Prediction Head (MLP):
$$\hat{y} = \mathbf{w}_{\text{out}}^T \mathbf{h}_G + b_{\text{out}}$$
""")

    add_code(nb, """class MolecularGCN:
    \"\"\"Two-layer Graph Convolutional Network implemented in pure NumPy.\"\"\"
    def __init__(self, in_dim=20, hidden_dim=32, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier / Glorot initialization
        scale0 = np.sqrt(2.0 / (in_dim + hidden_dim))
        self.W0 = rng.normal(0, scale0, (in_dim, hidden_dim)).astype(np.float32)
        self.b0 = np.zeros(hidden_dim, dtype=np.float32)

        scale1 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W1 = rng.normal(0, scale1, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        scale_out = np.sqrt(2.0 / hidden_dim)
        self.W_out = rng.normal(0, scale_out, (hidden_dim, 1)).astype(np.float32)
        self.b_out = 0.0

    def forward(self, H: np.ndarray, A_hat: np.ndarray) -> tuple:
        \"\"\"Forward pass returning predicted property and intermediate atom representations.\"\"\"
        # Layer 1: Message Passing + Linear + ReLU
        # Shape: (N, in_dim) -> (N, hidden_dim)
        Z0 = A_hat @ H @ self.W0 + self.b0
        H1 = np.maximum(0, Z0)  # ReLU

        # Layer 2: Message Passing + Linear + ReLU
        # Shape: (N, hidden_dim) -> (N, hidden_dim)
        Z1 = A_hat @ H1 @ self.W1 + self.b1
        H2 = np.maximum(0, Z1)  # ReLU

        # Global Readout: Mean Pooling across all atoms
        # Shape: (hidden_dim,)
        h_graph = np.mean(H2, axis=0)

        # Output MLP: Linear scalar regression
        y_pred = float(h_graph @ self.W_out + self.b_out)
        return y_pred, h_graph, H2

gcn_demo = MolecularGCN()
pred_val, h_mol, h_atoms = gcn_demo.forward(H_asp, A_hat_asp)
print(f"Forward pass successful:")
print(f"  • Whole-molecule graph representation vector: shape {h_mol.shape}")
print(f"  • Predicted property output: {pred_val:.3f}")
""")

    add_md(nb, """## 5. Solubility Regression: Delaney ESOL Dataset

We train our model to predict **aqueous solubility ($\log S$, in $\text{mol/L}$)**:
- $\log S > 0$: Highly soluble.
- $-2 < \log S \le 0$: Moderately soluble.
- $\log S \le -4$: Poorly soluble / lipophilic (formulation challenge).
""")

    add_code(nb, """# Curated Delaney ESOL subset
esol_data = [
    ("Methane", "C", -0.77), ("Ethane", "CC", -1.33), ("Propane", "CCC", -2.14),
    ("Butane", "CCCC", -2.71), ("Pentane", "CCCCC", -3.42), ("Hexane", "CCCCCC", -3.99),
    ("Benzene", "c1ccccc1", -2.13), ("Toluene", "Cc1ccccc1", -2.59), ("Phenol", "c1ccccc1O", -0.46),
    ("Aniline", "c1ccccc1N", -0.52), ("Benzoic acid", "c1ccccc1C(=O)O", -1.48),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", -1.61), ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", -0.84),
    ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C", -1.02), ("Ethanol", "CCO", 0.52),
    ("Methanol", "CO", 0.85), ("Acetic acid", "CC(=O)O", 0.73), ("Acetone", "CC(=O)C", 0.45),
    ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O", 0.38), ("Chlorobenzene", "c1ccc(Cl)cc1", -2.54),
    ("Nitrobenzene", "c1ccc(cc1)[N+](=O)[O-]", -1.78), ("Pyridine", "c1ccncc1", 0.12),
    ("Naphthalene", "c1ccc2ccccc2c1", -3.68), ("Anthracene", "c1ccc2cc3ccccc3cc2c1", -5.20),
    ("Donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", -4.50),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.20), ("Diclofenac", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", -4.10)
]

df_esol = pd.DataFrame(esol_data, columns=["name", "smiles", "logS_obs"])
print(f"Loaded {len(df_esol)} molecules from Delaney ESOL solubility dataset.")
df_esol.head(4)
""")

    add_code(nb, """# Train/Test Split: Scaffold split
train_idx, test_idx = metrics.scaffold_split(df_esol["smiles"].tolist(), train_ratio=0.75, seed=42)

df_esol_train = df_esol.iloc[train_idx].reset_index(drop=True)
df_esol_test = df_esol.iloc[test_idx].reset_index(drop=True)

# Train Classical Random Forest Baseline on ECFP4 fingerprints
def get_ecfp_matrix(smiles_list):
    arrs = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
        a = np.zeros(1024, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, a)
        arrs.append(a)
    return np.array(arrs)

X_train_rf = get_ecfp_matrix(df_esol_train["smiles"])
y_train_rf = df_esol_train["logS_obs"].values
X_test_rf = get_ecfp_matrix(df_esol_test["smiles"])
y_test_rf = df_esol_test["logS_obs"].values

rf_reg = ensemble.RandomForestRegressor(n_estimators=50, random_state=42)
rf_reg.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_reg.predict(X_test_rf)

rmse_rf = np.sqrt(skmetrics.mean_squared_error(y_test_rf, y_pred_rf))
r2_rf = skmetrics.r2_score(y_test_rf, y_pred_rf)
print(f"Random Forest + ECFP4 Baseline Test Performance:")
print(f"  • RMSE: {rmse_rf:.3f} logS")
print(f"  • R²:   {r2_rf:.3f}")
""")

    add_md(nb, """## 6. Guided Student Exercises & Self-Checks

---

### Exercise 1: Molecular Graph Adjacency Normalization
Implement `compute_normalized_adjacency(mol: Chem.Mol) -> np.ndarray`:
- Computes standard adjacency matrix $\mathbf{A}$.
- Adds self-loops: $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$.
- Computes degree vector $\mathbf{\tilde{d}} = \sum_j \tilde{A}_{ij}$.
- Returns symmetrically normalized adjacency:
  $$\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2}$$
""")

    add_code(nb, """# Exercise 1: Symmetrically Normalized Adjacency
def compute_normalized_adjacency(mol: Chem.Mol) -> np.ndarray:
    \"\"\"Compute symmetrically normalized adjacency matrix A_hat with self-loops.\"\"\"
    # --- SOLUTION BEGIN ---
    n = mol.GetNumHeavyAtoms()
    A = Chem.GetAdjacencyMatrix(mol)
    A_tilde = A + np.eye(n, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return np.array(A_hat, dtype=np.float32)
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 1
# Ethane: CC (2 atoms, bonded to each other)
# A_tilde = [[1, 1], [1, 1]], degrees = [2, 2]
# A_hat = [[0.5, 0.5], [0.5, 0.5]]
mol_ethane = Chem.MolFromSmiles("CC")
a_hat_ethane = compute_normalized_adjacency(mol_ethane)

np.testing.assert_allclose(a_hat_ethane, np.array([[0.5, 0.5], [0.5, 0.5]]), atol=1e-4)
# Self-loops ensure diagonal is non-zero
assert np.all(np.diag(a_hat_ethane) > 0.0)
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Graph Convolutional Layer Forward Pass
Implement `graph_conv_layer(H: np.ndarray, A_hat: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray`:
- Performs neighborhood aggregation and linear projection:
  $$\mathbf{Z} = \mathbf{\hat{A}} \mathbf{H} \mathbf{W} + \mathbf{b}$$
- Applies ReLU non-linear activation: $\text{ReLU}(z) = \max(0, z)$.
- Returns output feature matrix $\mathbf{H}' \in \mathbb{R}^{N \times d_{\text{out}}}$.
""")

    add_code(nb, """# Exercise 2: Graph Convolution Layer
def graph_conv_layer(H: np.ndarray, A_hat: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"Forward pass of a Graph Convolutional Layer with ReLU activation.\"\"\"
    # --- SOLUTION BEGIN ---
    aggregated = A_hat @ H
    projected = aggregated @ W + b
    H_out = np.maximum(0.0, projected)
    return H_out
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
N_test = 3
d_in = 4
d_out = 2

H_test = np.ones((N_test, d_in), dtype=np.float32)
A_test = np.eye(N_test, dtype=np.float32)  # Identity adjacency
W_test = np.full((d_in, d_out), 0.5, dtype=np.float32)
b_test = np.full(d_out, -1.0, dtype=np.float32)

# H @ W = 4 * 0.5 = 2.0; + b = 2.0 - 1.0 = 1.0; ReLU(1.0) = 1.0
out_test = graph_conv_layer(H_test, A_test, W_test, b_test)
np.testing.assert_allclose(out_test, np.ones((N_test, d_out)), atol=1e-4)
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: Global Mean Pooling Readout
Implement `global_mean_pool(atom_features: np.ndarray) -> np.ndarray`:
- Takes node representation matrix $\mathbf{H} \in \mathbb{R}^{N \times d}$.
- Averages across all $N$ atoms to produce a whole-molecule embedding vector:
  $$\mathbf{h}_G = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i$$
- Returns 1D array of shape $(d,)$.
""")

    add_code(nb, """# Exercise 3: Global Mean Pooling
def global_mean_pool(atom_features: np.ndarray) -> np.ndarray:
    \"\"\"Aggregate atom embeddings into a whole-molecule embedding via mean pooling.\"\"\"
    # --- SOLUTION BEGIN ---
    H = np.asarray(atom_features, dtype=np.float32)
    return np.mean(H, axis=0)
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
test_node_reps = np.array([
    [1.0, 2.0, 3.0],
    [3.0, 4.0, 5.0]
], dtype=np.float32)

mol_vec = global_mean_pool(test_node_reps)
np.testing.assert_allclose(mol_vec, [2.0, 3.0, 4.0], atol=1e-4)
assert mol_vec.ndim == 1, "Output must be a 1D vector"
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """---

### Exercise 4: Regression Error Evaluation (RMSE & R²)
Implement `compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict`:
- Computes Root Mean Squared Error (RMSE):
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- Computes Coefficient of Determination ($R^2$):
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- Returns dictionary with keys: `"RMSE"` and `"R2"`.
""")

    add_code(nb, """# Exercise 4: Regression Metrics
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    \"\"\"Compute RMSE and R2 between true and predicted property values.\"\"\"
    # --- SOLUTION BEGIN ---
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    
    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        "RMSE": float(round(rmse, 3)),
        "R2": float(round(r2, 3))
    }
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 4
metrics_rf = compute_regression_metrics(y_test_rf, y_pred_rf)
print(f"Calculated Regression Metrics: {metrics_rf}")
assert "RMSE" in metrics_rf and "R2" in metrics_rf
assert abs(metrics_rf["RMSE"] - rmse_rf) < 1e-2
print("✔ Exercise 4 passed!")
""")

    add_code(nb, """# Final validation
print("✔ NB21 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb21_gnn_property_prediction.ipynb",
        "solutions/nb21_gnn_property_prediction.ipynb"
    )

if __name__ == "__main__":
    build_nb21()
