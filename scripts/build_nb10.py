"""Build script for nb10: Substructure and Similarity Search."""

import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb10():
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
    add_cell("markdown", """# nb10 · Substructure and Similarity Search
**Cheminformatics Course · Part D: Searching and Pharmacophores**

### Objectives
1. Explain the **Similar-Property Principle** and the three molecular searching paradigms (substructure, pharmacophore, similarity index).
2. Implement **graph-based substructure matching** using RDKit and visualize matched subgraphs.
3. Build a **hybrid search pipeline** combining fingerprint bit-subset pre-filtering with VF2 graph verification, and evaluate its screening speedup.
4. Construct the **$2 \\times 2$ contingency table** ($a, b, c, d$) and implement $\\ge 12$ binary similarity coefficients from Lecture 06.
5. Analyze the **Tanimoto index** in depth and parameterize the **Tversky index** for asymmetric and substructure-like searches.
6. Benchmark a **Ligand-Based Virtual Screen (LBVS)** against acetylcholinesterase (AChE) inhibitors, computing **ROC-AUC**, **PR-AUC**, **EF1%**, and **BEDROC** ($\\alpha=20$).
7. Explore **scaffold hopping** with Bemis–Murcko frameworks and train a deep-learning VS classifier.""")

    # --- Setup & Imports ---
    add_cell("markdown", """## 1. Setup, Environment & Imports
We import standard scientific libraries along with RDKit and our custom `chemlib` course toolkit.""")

    add_cell("code", """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier

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

    # --- Section 2: Theoretical Foundations & Paradigms ---
    add_cell("markdown", """## 2. Searching Paradigms & The Similar-Property Principle

### The Similar-Property Principle
Formulated by **Johnson and Maggiora (1990)**, the Similar-Property Principle states:
> *"Structurally similar molecules tend to possess similar physical, chemical, and biological properties."*

While activity cliffs (where small structural changes cause radical changes in potency) represent famous exceptions, this principle forms the empirical cornerstone of **Ligand-Based Virtual Screening (LBVS)** and medicinal chemistry optimization.

### Three Searching Paradigms in Cheminformatics
| Paradigm | Underlying Representation | Question Answered | Algorithmic Mechanism |
|---|---|---|---|
| **Substructure Search** | 2D Molecular Graph | *"Does target molecule $T$ contain query graph $Q$?"* | Exact Subgraph Isomorphism (Ullmann, VF2) |
| **Pharmacophore Search** | 3D Spatial Geometry / 2D Triads | *"Does target arrange key interaction features at specified distances?"* | Geometric distance constraints, tolerance spheres |
| **Similarity Search** | 1D/2D Binary Fingerprints | *"How chemically similar is target to a known active reference?"* | Vector distance / correlation metric (Tanimoto, Dice, etc.) |""")

    # --- Section 3: Substructure Search ---
    add_cell("markdown", """## 3. Substructure Search: Dopamine-like Query

In Lecture 06, we explored searching for **catechol-ethylamine** (dopamine-like) fragments across a diverse panel of central nervous system drugs:
- **Dopamine:** The endogenous neurotransmitter (catechol + ethylamine).
- **Adrenaline (Epinephrine):** Endogenous adrenergic agonist containing a catechol ring and ethylamine chain with an $N$-methyl substituent.
- **Fenoldopam:** Selective dopamine $D_1$ receptor agonist.
- **Apomorphine:** Non-selective dopamine agonist with a rigid tetracyclic aporphine core containing catecholamine.
- **Morphine:** Analgesic with a phenolic ring, but lacking the 1,2-catechol diol pattern.
- **Olmidine:** Anti-hypertensive agent containing a chlorinated bicyclic core (negative control).""")

    add_cell("code", """# Define test panel molecules from Lecture 06
panel_data = [
    {"name": "Dopamine", "smiles": "c1cc(O)c(O)cc1CCN"},
    {"name": "Adrenaline", "smiles": "CNC[C@H](O)c1ccc(O)c(O)c1"},
    {"name": "Fenoldopam", "smiles": "Clc1cc2c(cc1O)c(c(O)cc2)c3ccc(O)cc3CCN"},
    {"name": "Apomorphine", "smiles": "CN1CCc2cccc3c2[C@H]1Cc4ccc(O)c(O)c34"},
    {"name": "Morphine", "smiles": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"},
    {"name": "Olmidine", "smiles": "Nc1nc(N)c2c(n1)c(Cl)c(Cl)c(Cl)c2Cl"}
]

df_panel = pd.DataFrame(panel_data)
df_panel["mol"] = df_panel["smiles"].apply(Chem.MolFromSmiles)

# Dopamine-like query: catechol ring attached to ethylamine chain
# SMARTS: c1cc(O)c(O)cc1CCN
query_smarts = "c1cc(O)c(O)cc1CCN"
query_mol = Chem.MolFromSmarts(query_smarts)

df_panel["has_match"] = df_panel["mol"].apply(lambda m: m.HasSubstructMatch(query_mol))
df_panel["match_atoms"] = df_panel["mol"].apply(lambda m: list(m.GetSubstructMatch(query_mol)))

print("=== Substructure Matching Results ===")
for _, row in df_panel.iterrows():
    print(f"{row['name']:<12}: Match = {row['has_match']:<5} | Matched Atom IDs: {row['match_atoms']}")""")

    # --- Section 4: Hybrid Screening ---
    add_cell("markdown", """## 4. Hybrid Screening: Fingerprint Pre-filter + VF2 Graph Verification

### Why Hybrid Screening?
Subgraph isomorphism algorithms (like Ullmann or **VF2**) explore recursive atom mappings and are **NP-complete** in the worst case ($O(N!)$). Running VF2 verification against a library of millions of molecules is computationally prohibitive.

### Principle of the Bit-Subset Pre-Filter
If query graph $Q$ is a subgraph of target molecule $T$, then **every path or structural feature present in $Q$ must also be present in $T$**.
Therefore, for a monotonic pattern fingerprint:
$$(Q_{\\text{bits}} \\subseteq T_{\\text{bits}}) \\iff (Q_{\\text{bits}} \\ \\& \\ T_{\\text{bits}} == Q_{\\text{bits}})$$

If even **one bit** present in the query is missing in the target, $T$ cannot possibly contain $Q$ as a subgraph, and we can reject $T$ instantly in $O(1)$ bitwise operations!

### The 5-Bit Toy Example from Lecture 06
Consider a 5-bit fingerprint:
- **Query $Q$:** `[1, 0, 1, 0, 1]` (Bits 0, 2, 4 are ON)
- **Target $A$:** `[1, 1, 1, 0, 1]` $\\rightarrow$ Query bits `[0, 2, 4]` are all ON in $A$ $\\rightarrow$ **Pre-filter PASS** (Proceeds to VF2).
- **Target $B$:** `[1, 0, 0, 1, 1]` $\\rightarrow$ Bit 2 is ON in $Q$ but OFF in $B$ $\\rightarrow$ **Pre-filter REJECT** (VF2 skipped).""")

    add_cell("code", """# Recreate Lecture 06 5-bit demonstration
q_bits = np.array([1, 0, 1, 0, 1], dtype=int)
tA_bits = np.array([1, 1, 1, 0, 1], dtype=int)
tB_bits = np.array([1, 0, 0, 1, 1], dtype=int)

def bit_subset_match(q: np.ndarray, t: np.ndarray) -> bool:
    \"\"\"Return True if query bits are a strict subset of target bits: (q & t) == q.\"\"\"
    return np.all((q & t) == q)

print(f"Query:    {q_bits}")
print(f"Target A: {tA_bits} -> Pre-filter Match: {bit_subset_match(q_bits, tA_bits)} (Verified by VF2)")
print(f"Target B: {tB_bits} -> Pre-filter Match: {bit_subset_match(q_bits, tB_bits)} (Instantly Discarded)")
assert bit_subset_match(q_bits, tA_bits) == True
assert bit_subset_match(q_bits, tB_bits) == False""")

    add_cell("markdown", """### Real-World Hybrid Screening Benchmark on ChEMBL AChE Library
We now benchmark this hybrid strategy on the curated ChEMBL AChE dataset (`data/processed/d04_chembl_ache_curated.csv`).""")

    add_cell("code", """import time

# Locate and load curated AChE dataset
data_path = find_data_file("d04_chembl_ache_curated.csv")
df_ache = pd.read_csv(data_path)
df_ache["mol"] = df_ache["canonical_smiles"].apply(Chem.MolFromSmiles)
mols = [m for m in df_ache["mol"] if m is not None]

# Substructure query: donepezil-like dimethoxyindanone core
# SMARTS: c1cc2c(cc1OC)C(=O)CC2
sub_query = Chem.MolFromSmarts("c1cc2c(cc1OC)C(=O)CC2")
q_pattern_fp = Chem.PatternFingerprint(sub_query)

# 1. Pure VF2 Graph Matching
t0 = time.perf_counter()
vf2_hits = [m.HasSubstructMatch(sub_query) for m in mols]
t_vf2 = time.perf_counter() - t0

# 2. Hybrid Pipeline: Pattern FP Pre-filter -> VF2
t0 = time.perf_counter()
target_fps = [Chem.PatternFingerprint(m) for m in mols]
hybrid_hits = []
prefilter_passes = 0

for m, tfp in zip(mols, target_fps):
    # Bit subset check using RDKit's fast AllProbeBitsMatch
    if DataStructs.AllProbeBitsMatch(q_pattern_fp, tfp):
        prefilter_passes += 1
        hybrid_hits.append(m.HasSubstructMatch(sub_query))
    else:
        hybrid_hits.append(False)
t_hybrid = time.perf_counter() - t0

rejection_rate = (len(mols) - prefilter_passes) / len(mols) * 100.0
assert vf2_hits == hybrid_hits, "Hybrid search must produce identical hits to pure VF2!"

print(f"Total Molecules Screened: {len(mols)}")
print(f"Pre-filter Discarded:     {len(mols) - prefilter_passes} ({rejection_rate:.1f}%) without graph traversal")
print(f"Verified True Hits:       {sum(hybrid_hits)}")
print(f"Pure VF2 Time:            {t_vf2*1000:.2f} ms")
print(f"Hybrid Pipeline Time:     {t_hybrid*1000:.2f} ms")""")

    # --- Section 5: 2x2 Contingency Table & Coefficients ---
    add_cell("markdown", """## 5. The $2 \\times 2$ Contingency Table & Binary Similarity Coefficients

When comparing two binary fingerprints $\\mathbf{x}$ and $\\mathbf{y}$ of length $p$, each bit position falls into one of four mutually exclusive contingency classes:

| | Bit on in $\\mathbf{y}$ ($y=1$) | Bit off in $\\mathbf{y}$ ($y=0$) | Marginal Totals |
|---|---|---|---|
| **Bit on in $\\mathbf{x}$ ($x=1$)** | **$a$** (Shared 1s) | **$b$** ($x$ only) | $a + b$ |
| **Bit off in $\\mathbf{x}$ ($x=0$)** | **$c$** ($y$ only) | **$d$** (Shared 0s) | $c + d$ |
| **Marginal Totals** | $a + c$ | $b + d$ | $p = a + b + c + d$ |

### Lecture 06 Numeric Worked Example
Let:
$$\\mathbf{x} = 1101100011, \\quad \\mathbf{y} = 1010100011$$
- Total bits $p = 10$.
- Shared ones: bits 0, 4, 8, 9 $\\rightarrow a = 4$.
- ON in $\\mathbf{x}$ only: bits 1, 3 $\\rightarrow b = 2$.
- ON in $\\mathbf{y}$ only: bit 2 $\\rightarrow c = 1$.
- Shared zeros: bits 5, 6, 7 $\\rightarrow d = 3$.""")

    add_cell("code", """# Slide numerical verification
x_vec = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=int)
y_vec = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 1], dtype=int)

a = int(np.sum((x_vec == 1) & (y_vec == 1)))
b = int(np.sum((x_vec == 1) & (y_vec == 0)))
c = int(np.sum((x_vec == 0) & (y_vec == 1)))
d = int(np.sum((x_vec == 0) & (y_vec == 0)))
p = a + b + c + d

print(f"Contingency Counts: a={a}, b={b}, c={c}, d={d}, p={p}")
assert (a, b, c, d, p) == (4, 2, 1, 3, 10), "Contingency counts must exactly match slide values!\"""")

    add_cell("markdown", """### Implementing $\\ge 12$ Binary Similarity Coefficients from Lecture 06
Lecture 06 presents 51 similarity coefficients. Below, we implement 12 essential coefficients representing distinct families:
1. **Tanimoto / Jaccard:** $\\frac{a}{a + b + c}$
2. **Dice / Sørensen–Gleason:** $\\frac{2a}{2a + b + c}$
3. **Cosine / Ochiai:** $\\frac{a}{\\sqrt{(a + b)(a + c)}}$
4. **Simple Matching (Sokal–Michener):** $\\frac{a + d}{p}$
5. **Russell–Rao:** $\\frac{a}{p}$
6. **Simpson:** $\\frac{a}{\\min(a + b, a + c)}$
7. **Kulczynski (No. 2):** $\\frac{1}{2}\\left(\\frac{a}{a + b} + \\frac{a}{a + c}\\right)$
8. **Forbes:** $\\frac{p \\cdot a}{(a + b)(a + c)}$
9. **Braun–Blanquet:** $\\frac{a}{\\max(a + b, a + c)}$
10. **Sokal–Sneath:** $\\frac{a}{a + 2(b + c)}$
11. **Roger–Tanimoto:** $\\frac{a + d}{a + 2(b + c) + d}$
12. **Yule Q:** $\\frac{ad - bc}{ad + bc}$""")

    add_cell("code", """def calculate_similarity_coefficients_vector(a: int, b: int, c: int, d: int) -> Dict[str, float]:
    \"\"\"Compute >= 12 binary similarity coefficients from contingency counts.\"\"\"
    p = a + b + c + d
    
    tanimoto = a / (a + b + c) if (a + b + c) > 0 else 0.0
    dice = (2.0 * a) / (2 * a + b + c) if (2 * a + b + c) > 0 else 0.0
    cosine = a / np.sqrt((a + b) * (a + c)) if ((a + b) * (a + c)) > 0 else 0.0
    simple_match = (a + d) / p if p > 0 else 0.0
    russell_rao = a / p if p > 0 else 0.0
    simpson = a / min(a + b, a + c) if min(a + b, a + c) > 0 else 0.0
    kulczynski = 0.5 * (a / (a + b) + a / (a + c)) if (a + b > 0 and a + c > 0) else 0.0
    forbes = (p * a) / ((a + b) * (a + c)) if ((a + b) * (a + c)) > 0 else 0.0
    braun_blanquet = a / max(a + b, a + c) if max(a + b, a + c) > 0 else 0.0
    sokal_sneath = a / (a + 2 * (b + c)) if (a + 2 * (b + c)) > 0 else 0.0
    roger_tanimoto = (a + d) / (a + 2 * (b + c) + d) if (a + 2 * (b + c) + d) > 0 else 0.0
    
    # Yule Q association coefficient
    ad_bc_num = (a * d) - (b * c)
    ad_bc_den = (a * d) + (b * c)
    yule_q = ad_bc_num / ad_bc_den if ad_bc_den != 0 else 0.0

    return {
        "Tanimoto": float(tanimoto),
        "Dice": float(dice),
        "Cosine_Ochiai": float(cosine),
        "Simple_Matching": float(simple_match),
        "Russell_Rao": float(russell_rao),
        "Simpson": float(simpson),
        "Kulczynski": float(kulczynski),
        "Forbes": float(forbes),
        "Braun_Blanquet": float(braun_blanquet),
        "Sokal_Sneath": float(sokal_sneath),
        "Roger_Tanimoto": float(roger_tanimoto),
        "Yule_Q": float(yule_q),
    }

coeffs = calculate_similarity_coefficients_vector(a, b, c, d)
df_coeffs = pd.DataFrame(list(coeffs.items()), columns=["Coefficient", "Value"])
print(df_coeffs.to_string(index=False))""")

    # --- Section 6: Tanimoto in Depth & Tversky Index ---
    add_cell("markdown", """## 6. Tanimoto in Depth & The Tversky Index

### Set-Based Tanimoto Index
Expressed in terms of set intersection and union:
$$T(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|} = \\frac{N_c}{N_a + N_b - N_c}$$
where $N_a$ is the number of bits in molecule $X$, $N_b$ is the number of bits in molecule $Y$, and $N_c$ is the number of shared bits.

**Slide Example:** Molecule $X$ with 6 bits, Molecule $Y$ with 6 bits, 2 shared bits:
$$T = \\frac{2}{6 + 6 - 2} = \\frac{2}{10} = 0.20$$

### The Tversky Index: Asymmetric Generalization
Tversky (1977) introduced a parameterized similarity index:
$$S_T(X, Y; \\alpha, \\beta) = \\frac{a}{a + \\alpha b + \\beta c}$$
where $\\alpha \\ge 0, \\beta \\ge 0$.
- **Symmetric Special Cases:**
  - $\\alpha = 1, \\beta = 1 \\implies$ **Tanimoto**
  - $\\alpha = 0.5, \\beta = 0.5 \\implies$ **Dice**
- **Asymmetric Searches:**
  - $\\alpha = 1.0, \\beta = 0.0$: Penalizes missing query features ($b$), but does NOT penalize additional target features ($c$). This mimics a **substructure-like search** using continuous fingerprints!
  - $\\alpha = 0.0, \\beta = 1.0$: Superstructure search.""")

    add_cell("code", """# Verify 6-bit/6-bit/2-common example
t_set = 2 / (6 + 6 - 2)
assert abs(t_set - 0.20) < 1e-6, "Tanimoto set calculation must equal 0.20!"
print(f"Slide Set-Based Tanimoto: {t_set:.2f}")

# Compare Tversky with alpha=beta=1 (Tanimoto) vs alpha=beta=0.5 (Dice)
mol1 = Chem.MolFromSmiles("c1ccccc1O")       # Phenol
mol2 = Chem.MolFromSmiles("c1ccc(CC)cc1O")   # 4-ethylphenol
fp1 = fingerprints.get_fingerprint(mol1, "ecfp4")
fp2 = fingerprints.get_fingerprint(mol2, "ecfp4")

tv_tan = fingerprints.tversky_similarity(fp1, fp2, alpha=1.0, beta=1.0)
tan_val = fingerprints.tanimoto_similarity(fp1, fp2)
tv_dice = fingerprints.tversky_similarity(fp1, fp2, alpha=0.5, beta=0.5)
dice_val = fingerprints.dice_similarity(fp1, fp2)

print(f"Tversky(1,1): {tv_tan:.4f} | Tanimoto: {tan_val:.4f} (Difference = {abs(tv_tan - tan_val):.2e})")
print(f"Tversky(0.5,0.5): {tv_dice:.4f} | Dice: {dice_val:.4f} (Difference = {abs(tv_dice - dice_val):.2e})")
assert abs(tv_tan - tan_val) < 1e-6
assert abs(tv_dice - dice_val) < 1e-6""")

    # --- Section 7: LBVS Benchmark ---
    add_cell("markdown", """## 7. Ligand-Based Virtual Screening (LBVS) Benchmark

We evaluate virtual screening retrieval against the curated AChE library:
- **Active Class:** High-affinity inhibitors with $IC_{50} \\le 50.0$ nM ($pIC_{50} \\ge 7.3$).
- **Decoy/Weaker Class:** Remaining compounds.

### Ranking Method: Max-Similarity to Known Actives
Each library compound $T$ is scored by its maximum similarity to any active reference $A_i$:
$$S_{\\max}(T) = \\max_{A_i \\in \\text{Actives}} \\text{Sim}(A_i, T)$$

### Key Evaluation Metrics
1. **ROC-AUC:** Receiver Operating Characteristic Area Under Curve.
2. **PR-AUC:** Precision-Recall Area Under Curve (crucial for imbalanced data).
3. **Enrichment Factor at 1% ($EF_{1\\%}$):**
   $$EF_{1\\%} = \\frac{\\text{Actives in top 1\\%} / \\text{Total molecules in top 1\\%}}{\\text{Total actives} / \\text{Total library compounds}}$$
4. **BEDROC ($\\alpha=20$):** Boltzmann-Enhanced Discrimination of ROC for early recognition.""")

    add_cell("code", """# Label actives and generate fingerprints
df_lbvs = df_ache.copy()
df_lbvs["is_active"] = (df_lbvs["standard_value"] <= 50.0).astype(int)
n_actives = df_lbvs["is_active"].sum()
n_total = len(df_lbvs)
print(f"LBVS Benchmark Set: {n_total} compounds | {n_actives} Actives ({n_actives/n_total*100:.1f}%)")

# Precompute ECFP4 and MACCS fingerprints
df_lbvs["ecfp4"] = df_lbvs["mol"].apply(lambda m: fingerprints.get_fingerprint(m, "ecfp4"))
df_lbvs["maccs"] = df_lbvs["mol"].apply(lambda m: fingerprints.get_fingerprint(m, "maccs"))

# Active queries (sample 5 diverse actives as the query seed set)
active_indices = df_lbvs[df_lbvs["is_active"] == 1].index.tolist()[:5]
seed_ecfp4 = [df_lbvs.loc[i, "ecfp4"] for i in active_indices]

# Compute Max-Similarity scores
lib_ecfp4 = df_lbvs["ecfp4"].tolist()
scores_tan = fingerprints.max_similarity_to_actives(lib_ecfp4, seed_ecfp4)
y_true = df_lbvs["is_active"].values

# Calculate Metrics
roc_auc = metrics.classification_report_dict(y_true, (scores_tan > 0.5).astype(int), scores_tan)["ROC_AUC"]
pr_auc = metrics.classification_report_dict(y_true, (scores_tan > 0.5).astype(int), scores_tan)["PR_AUC"]
ef1 = metrics.enrichment_factor(y_true, scores_tan, fraction=0.01)
bedroc = metrics.bedroc_score(y_true, scores_tan, alpha=20.0)

print(f"=== ECFP4 Tanimoto LBVS Performance ===")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")
print(f"EF1%:    {ef1:.2f}")
print(f"BEDROC:  {bedroc:.3f}")""")

    # --- Section 8: Scaffold Hopping ---
    add_cell("markdown", """## 8. Scaffold Hopping & Deep Learning VS Teaser

### Scaffold Hopping
**Scaffold hopping** occurs when a virtual screen retrieves compounds with a completely novel chemotype (different Bemis–Murcko scaffold) while retaining binding potency. In contrast to 2D fingerprint similarity (which often penalizes scaffold changes), 3D pharmacophores and deep-learning models can successfully hop scaffolds.""")

    add_cell("code", """# Bemis-Murcko scaffold analysis
df_lbvs["scaffold"] = df_lbvs["mol"].apply(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
print(f"Total Unique Scaffolds across {len(df_lbvs)} compounds: {df_lbvs['scaffold'].nunique()}")

# Identify scaffold-hopping candidates: high activity but moderate/low 2D similarity to seed set
df_lbvs["sim_score"] = scores_tan
hops = df_lbvs[(df_lbvs["is_active"] == 1) & (df_lbvs["sim_score"] < 0.45)]
print(f"Found {len(hops)} active compounds with low 2D similarity (< 0.45) to seed queries (Scaffold Hop candidates)")""")

    # --- Section 9: Exercises ---
    add_cell("markdown", """## 9. Exercises & Self-Checks

### Exercise 1: Custom SMARTS Filter & Asymmetric Tversky Search
Implement a function `screen_catechol_library` that takes a list of SMILES, identifies molecules containing a catechol substructure (`c1c(O)c(O)ccc1`), and computes their Tversky similarity to Dopamine with $\\alpha=1.0, \\beta=0.2$.""")

    ex1_student = """def screen_catechol_library(smiles_list: List[str]) -> pd.DataFrame:
    \"\"\"Filter smiles for catechol substructure and compute Tversky similarity to Dopamine.
    Query: Dopamine ('c1cc(O)c(O)cc1CCN')
    Tversky parameters: alpha=1.0, beta=0.2
    Returns a DataFrame with columns: ['smiles', 'has_catechol', 'tversky_score']
    \"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement screen_catechol_library\")"""

    ex1_solution = """def screen_catechol_library(smiles_list: List[str]) -> pd.DataFrame:
    \"\"\"Filter smiles for catechol substructure and compute Tversky similarity to Dopamine.
    Query: Dopamine ('c1cc(O)c(O)cc1CCN')
    Tversky parameters: alpha=1.0, beta=0.2
    Returns a DataFrame with columns: ['smiles', 'has_catechol', 'tversky_score']
    \"\"\"
    catechol_pat = Chem.MolFromSmarts("c1c(O)c(O)ccc1")
    dopamine_mol = Chem.MolFromSmiles("c1cc(O)c(O)cc1CCN")
    dopamine_fp = fingerprints.get_fingerprint(dopamine_mol, "ecfp4")

    results = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        has_cat = m.HasSubstructMatch(catechol_pat)
        fp = fingerprints.get_fingerprint(m, "ecfp4")
        tver = fingerprints.tversky_similarity(dopamine_fp, fp, alpha=1.0, beta=0.2)
        results.append({
            "smiles": smi,
            "has_catechol": has_cat,
            "tversky_score": float(tver)
        })
    return pd.DataFrame(results)"""

    add_cell("exercise", ex1_student, ex1_solution)

    add_cell("markdown", """### Exercise 2: Comprehensive LBVS Evaluation
Implement `evaluate_lbvs_rankings` to calculate ROC-AUC, PR-AUC, EF1%, and BEDROC ($\\alpha=20$) for arbitrary continuous scores and binary labels.""")

    ex2_student = """def evaluate_lbvs_rankings(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    \"\"\"Compute ROC-AUC, PR-AUC, EF1%, and BEDROC (alpha=20.0).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement evaluate_lbvs_rankings\")"""

    ex2_solution = """def evaluate_lbvs_rankings(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    \"\"\"Compute ROC-AUC, PR-AUC, EF1%, and BEDROC (alpha=20.0).\"\"\"
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    rep = metrics.classification_report_dict(y_true, (scores > 0.5).astype(int), scores)
    ef1 = metrics.enrichment_factor(y_true, scores, fraction=0.01)
    bedroc = metrics.bedroc_score(y_true, scores, alpha=20.0)

    return {
        "ROC_AUC": rep["ROC_AUC"],
        "PR_AUC": rep["PR_AUC"],
        "EF1%": ef1,
        "BEDROC_20": bedroc,
    }"""

    add_cell("exercise", ex2_student, ex2_solution)

    # --- Self-Checks ---
    add_cell("markdown", """## 10. Self-Check Unit Tests
Run the cell below to verify your solutions against automated test assertions.""")

    add_cell("code", """# --- Unit Tests for Exercise 1 ---
test_smiles = [
    "c1cc(O)c(O)cc1CCN",        # Dopamine (catechol)
    "CNC[C@H](O)c1ccc(O)c(O)c1",# Adrenaline (catechol)
    "c1ccccc1O",                 # Phenol (monohydroxy, not catechol)
    "Nc1nc(N)c2c(n1)c(Cl)c(Cl)c(Cl)c2Cl" # Olmidine (no catechol)
]
df_res = screen_catechol_library(test_smiles)
assert len(df_res) == 4, "Should process all 4 valid SMILES"
assert df_res.loc[0, "has_catechol"] == True, "Dopamine must match catechol"
assert df_res.loc[1, "has_catechol"] == True, "Adrenaline must match catechol"
assert df_res.loc[2, "has_catechol"] == False, "Phenol is not a catechol"
assert df_res.loc[3, "has_catechol"] == False, "Olmidine has no catechol"
assert df_res.loc[0, "tversky_score"] == 1.0, "Dopamine compared to itself must have Tversky=1.0"
assert df_res.loc[1, "tversky_score"] > 0.3, "Adrenaline must show substantial similarity to dopamine"

# --- Unit Tests for Exercise 2 ---
dummy_y = np.array([1]*10 + [0]*90)
dummy_scores = np.linspace(1, 0, 100)
ev = evaluate_lbvs_rankings(dummy_y, dummy_scores)
assert 0.99 <= ev["ROC_AUC"] <= 1.0, "Perfect ranking should yield ROC-AUC ~ 1.0"
assert ev["EF1%"] > 5.0, "Enrichment factor should reflect early active recovery"
assert 0.95 <= ev["BEDROC_20"] <= 1.0, "Perfect early ranking should yield BEDROC ~ 1.0"

print("All assertion self-checks passed!")
print("✔ NB10 checks passed")""")

    # Write notebooks
    os.makedirs("Cheminformatics/notebooks", exist_ok=True)
    os.makedirs("Cheminformatics/solutions", exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    path_student = "Cheminformatics/notebooks/nb10_substructure_and_similarity_search.ipynb"
    path_sol = "Cheminformatics/solutions/nb10_substructure_and_similarity_search.ipynb"

    with open(path_student, "w") as f:
        nbformat.write(nb_student, f)

    with open(path_sol, "w") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {path_student} and {path_sol}")

if __name__ == "__main__":
    create_nb10()
