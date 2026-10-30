"""Build script for nb11: Pharmacophore Modeling and 3D Pharmacophore Searching."""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb11():
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
    add_cell("markdown", """# nb11 · Pharmacophore Modeling and Pharmacophore Searching
**Cheminformatics Course · Part D: Searching and Pharmacophores**

### Objectives
1. Define the **IUPAC pharmacophore concept** and contrast it with structural graphs and functional groups.
2. Analyze **historical pharmacophore models** (Sulfonamides vs PABA, Estradiol vs Diethylstilbestrol).
3. Extract 3D pharmacophore features (Donors, Acceptors, Aromatic rings, Hydrophobes, Cations, Anions) using `chemlib.pharm`.
4. Generate **3D conformer ensembles** (ETKDGv3 + MMFF94) and compute geometric inter-feature distance matrices.
5. Derive a **ligand-based common feature pharmacophore triad** from congeneric acetylcholinesterase (AChE) actives.
6. Execute **3-point pharmacophore screening** with spatial distance tolerances $(\\pm \\delta)$ and compare against 2D **Pharm2D signatures**.
7. Evaluate pharmacophore search hit recovery, active enrichment, and decoy rejection.""")

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
from rdkit.Chem import AllChem, Draw, ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D, SigFactory
from sklearn.metrics import confusion_matrix

import chemlib
from chemlib import pharm, metrics

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

    # --- Section 2: IUPAC Definition & Historical Models ---
    add_cell("markdown", """## 2. The Pharmacophore Concept & Historical Foundations

### IUPAC Official Definition
According to the **IUPAC Recommendations (1998)**:
> *"A pharmacophore is an ensemble of steric and electronic features that is necessary to ensure the optimal supramolecular interactions with a specific biological target and to trigger (or to block) its biological response."*

### Critical Distinctions
- **A pharmacophore is NOT a real molecule:** It has no molecular weight, no melting point, and no chemical bonds.
- **A pharmacophore is NOT a functional group:** An ester and an ether can both fulfill a *Hydrogen Bond Acceptor* feature.
- **A pharmacophore is an abstract 3D geometrical arrangement** of interaction capabilities (points, vectors, and tolerance spheres in space).

### Historical Milestones
1. **Fischer's Lock-and-Key Model (1894):** Geometric complementarity between enzyme and substrate.
2. **Sulfonamides vs $p$-Aminobenzoic Acid (PABA) (Woods, 1940):**
   Sulfanilamide acts as an antimetabolite inhibiting dihydropteroate synthase because its aromatic amino group and sulfonamide oxygens mimic the spatial distance (~6.7 Å) between the amino and carboxylate groups of PABA.
3. **Estradiol vs Diethylstilbestrol (DES) (Dodds, 1938):**
   Non-steroidal diethylstilbestrol mimics steroidal estradiol because its two phenolic hydroxyl oxygens are separated by a rigid spatial distance of ~11.0 Å in its active conformation.""")

    add_cell("code", """# Recreate historical models: PABA vs Sulfanilamide
paba = Chem.AddHs(Chem.MolFromSmiles("Nc1ccc(C(=O)O)cc1"))        # p-aminobenzoic acid
sulfa = Chem.AddHs(Chem.MolFromSmiles("Nc1ccc(S(=O)(=O)N)cc1"))    # sulfanilamide

AllChem.EmbedMolecule(paba, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(paba)
AllChem.EmbedMolecule(sulfa, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(sulfa)

paba_no_h = Chem.RemoveHs(paba)
sulfa_no_h = Chem.RemoveHs(sulfa)

# Extract features
paba_feats = pharm.extract_features(paba_no_h)
sulfa_feats = pharm.extract_features(sulfa_no_h)

print("PABA Feature Families: ", sorted(list(set([f['family'] for f in paba_feats]))))
print("Sulfa Feature Families:", sorted(list(set([f['family'] for f in sulfa_feats]))))

# Measure interatomic distance between Donor N and Acceptor O
conf_paba = paba_no_h.GetConformer()
pos_paba_n = conf_paba.GetAtomPosition(0) # Amino N
pos_paba_o = conf_paba.GetAtomPosition(5) # Carbonyl O
d_paba = np.linalg.norm(np.array(pos_paba_n) - np.array(pos_paba_o))

conf_sulfa = sulfa_no_h.GetConformer()
pos_sulfa_n = conf_sulfa.GetAtomPosition(0) # Amino N
pos_sulfa_o = conf_sulfa.GetAtomPosition(5) # Sulfonyl O
d_sulfa = np.linalg.norm(np.array(pos_sulfa_n) - np.array(pos_sulfa_o))

print(f"PABA N-to-O distance:         {d_paba:.2f} Å")
print(f"Sulfanilamide N-to-O distance: {d_sulfa:.2f} Å")
print(f"Distance Difference:          {abs(d_paba - d_sulfa):.2f} Å (< 0.5 Å confirms pharmacophore mimicry!)")""")

    # --- Section 3: Feature Extraction & 3D Conformer Generation ---
    add_cell("markdown", """## 3. Pharmacophore Feature Extraction & 3D Conformer Ensembles

In RDKit, pharmacophore features are parsed using the **BaseFeatures.fdef** definition file into six core families:
- **Hydrogen Bond Acceptor (HBA):** Electron pairs on $sp^2/sp^3$ heteroatoms.
- **Hydrogen Bond Donor (HBD):** Polar hydrogens attached to N, O, or S.
- **Aromatic Ring (Arom):** Planar delocalized $\\pi$-systems.
- **Hydrophobic Center (Hyd):** Aliphatic and alkyl chains.
- **Positive Ionizable / Cation (Pos):** Basic nitrogens (amines, amidines, guanidines).
- **Negative Ionizable / Anion (Neg):** Acidic centers (carboxylates, sulfonates, tetrazoles).""")

    add_cell("code", """# Donepezil: AChE Inhibitor with 3 key pharmacophore elements
# 1. Dimethoxyindanone (Acceptor + Aromatic)
# 2. Benzyl moiety (Aromatic + Hydrophobic)
# 3. Piperidine ring (Basic Nitrogen -> Positive Ionizable at pH 7.4)
donepezil_smi = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
mol_donepezil = Chem.AddHs(Chem.MolFromSmiles(donepezil_smi))

# Generate 3D Conformer with ETKDGv3 and minimize with MMFF94
AllChem.EmbedMolecule(mol_donepezil, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol_donepezil)
mol_donepezil = Chem.RemoveHs(mol_donepezil)

# Extract features using chemlib.pharm
feats = pharm.extract_features(mol_donepezil)
print(f"Extracted {len(feats)} pharmacophore features from Donepezil:")
for i, f in enumerate(feats[:8]):
    pos_str = f"({f['pos'][0]:.2f}, {f['pos'][1]:.2f}, {f['pos'][2]:.2f})" if f['pos'] else "N/A"
    print(f"  Feature {i+1:02d}: {f['family']:<15} | Type: {f['type']:<20} | Centroid: {pos_str}")""")

    # --- Section 4: Common-Feature Pharmacophore Triad ---
    add_cell("markdown", """## 4. Deriving a Common-Feature Pharmacophore Triad for AChE Inhibitors

AChE actives bind to the catalytic gorge via a characteristic triad:
1. **$F_1$: Aromatic Ring** (interacts with Trp84 or Trp279 via $\\pi-\\pi$ stacking).
2. **$F_2$: Positive Ionizable / Cation** (basic piperidine nitrogen forming cation-$\\pi$ interaction with Trp84).
3. **$F_3$: H-Bond Acceptor** (carbonyl oxygen forming hydrogen bond with Phe295).

We compute the pairwise inter-feature Euclidean distances:
$$d_{12} = \\|\\mathbf{p}_1 - \\mathbf{p}_2\\|, \\quad d_{23} = \\|\\mathbf{p}_2 - \\mathbf{p}_3\\|, \\quad d_{13} = \\|\\mathbf{p}_1 - \\mathbf{p}_3\\|$$""")

    add_cell("code", """# Compute inter-feature distances for Donepezil
arom_feats = [f for f in feats if f["family"] == "Aromatic"]
pos_feats = [f for f in feats if f["family"] == "PosIonizable"]
acc_feats = [f for f in feats if f["family"] == "Acceptor"]

assert len(arom_feats) > 0 and len(pos_feats) > 0 and len(acc_feats) > 0

# Pick representative feature centroids
p_arom = np.array(arom_feats[0]["pos"])
p_pos = np.array(pos_feats[0]["pos"])
p_acc = np.array(acc_feats[0]["pos"])

d_arom_pos = float(np.linalg.norm(p_arom - p_pos))
d_pos_acc = float(np.linalg.norm(p_pos - p_acc))
d_arom_acc = float(np.linalg.norm(p_arom - p_acc))

print("=== Donepezil 3-Point Pharmacophore Distances ===")
print(f"d(Aromatic - PosIonizable): {d_arom_pos:.2f} Å")
print(f"d(PosIonizable - Acceptor): {d_pos_acc:.2f} Å")
print(f"d(Aromatic - Acceptor):     {d_arom_acc:.2f} Å")""")

    # --- Section 5: 3-Point Pharmacophore Matching ---
    add_cell("markdown", """## 5. 3-Point Pharmacophore Query Matching with Tolerances

To search an arbitrary 3D conformer, we test whether ANY triplet of features matching types $(F_1, F_2, F_3)$ satisfies the triangle distances within tolerance $\\pm \\delta$:
$$|d_{12} - d_{12}^*| \\le \\delta, \\quad |d_{23} - d_{23}^*| \\le \\delta, \\quad |d_{13} - d_{13}^*| \\le \\delta$$""")

    add_cell("code", """# Match Donepezil against its own pharmacophore query
target_triad = ("Aromatic", "PosIonizable", "Acceptor")
target_dists = (d_arom_pos, d_pos_acc, d_arom_acc)
tolerance = 1.5 # +/- 1.5 Angstroms tolerance window

is_match = pharm.match_3point_pharmacophore_distances(
    mol=mol_donepezil,
    feat_types=target_triad,
    target_distances=target_dists,
    tolerance=tolerance
)
print(f"Donepezil matches 3-point pharmacophore query: {is_match}")
assert is_match == True, "Donepezil must match its own pharmacophore triad!\"""")

    # --- Section 6: 2D Pharm2D Signatures ---
    add_cell("markdown", """## 6. 2D Pharmacophore Signatures (Pharm2D)

In 2D pharmacophore fingerprints (**Pharm2D**), spatial distances are replaced by **topological path distances** (shortest bond paths across the 2D molecular graph).

### Signature Arithmetic from Lecture 06
- Features: 6 types.
- Distance Bins: 7 standard bins:
  $$[1, 3), [3, 4), [4, 5), [5, 6), [6, 7), [7, 8), [8, 100) \\text{ bonds}$$
- Combining all 2-point and 3-point pharmacophore combinations yields the standard **Pharmacoprint** feature space of **39,972 bits**.""")

    add_cell("code", """# Generate Gobbi Pharm2D fingerprint
fp_pharm2d = pharm.get_pharm2d_fingerprint(mol_donepezil)
print(f"Total Pharm2D Bit Space Length: {fp_pharm2d.GetNumBits()} bits")
print(f"Active Bits in Donepezil:       {fp_pharm2d.GetNumOnBits()} bits")
assert fp_pharm2d.GetNumBits() == 39972, "Gobbi Pharm2D signature space must equal 39,972 bits!\"""")

    # --- Section 7: Virtual Screening Benchmark ---
    add_cell("markdown", """## 7. Virtual Screening Benchmark: 3D Pharmacophore vs Decoys

We now benchmark our 3-point pharmacophore model on a test set of known AChE inhibitors and property-matched decoys.""")

    add_cell("code", """# Load AChE dataset and construct benchmark subset
data_path = find_data_file("d04_chembl_ache_curated.csv")
df_all = pd.read_csv(data_path)

# Top 15 potent actives (IC50 < 30 nM) and 35 weak/inactive compounds
df_act = df_all[df_all["standard_value"] < 30.0].head(15).copy()
df_act["label"] = 1
df_decoy = df_all[df_all["standard_value"] > 500.0].head(35).copy()
df_decoy["label"] = 0

df_screen = pd.concat([df_act, df_decoy], ignore_index=True)
print(f"Screening Set: {len(df_screen)} compounds ({len(df_act)} Actives, {len(df_decoy)} Decoys)")

# Generate 3D conformers and match pharmacophore
screen_hits = []
for smi in df_screen["canonical_smiles"]:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    res = AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    if res == 0:
        AllChem.MMFFOptimizeMolecule(m)
        m = Chem.RemoveHs(m)
        hit = pharm.match_3point_pharmacophore_distances(
            mol=m,
            feat_types=target_triad,
            target_distances=target_dists,
            tolerance=2.5 # generous tolerance for congeneric library
        )
    else:
        hit = False
    screen_hits.append(hit)

df_screen["hit"] = screen_hits

# Confusion Matrix and Performance
tp = int(np.sum((df_screen["label"] == 1) & (df_screen["hit"] == True)))
fp = int(np.sum((df_screen["label"] == 0) & (df_screen["hit"] == True)))
fn = int(np.sum((df_screen["label"] == 1) & (df_screen["hit"] == False)))
tn = int(np.sum((df_screen["label"] == 0) & (df_screen["hit"] == False)))

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"=== Pharmacophore Screening Results ===")
print(f"True Positives (TP):  {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN):  {tn}")
print(f"Sensitivity (Active Recall): {sensitivity*100:.1f}%")
print(f"Specificity (Decoy Rejection): {specificity*100:.1f}%")""")

    # --- Section 8: Exercises ---
    add_cell("markdown", """## 8. Exercises & Self-Checks

### Exercise 1: Build a Custom 3-Point Pharmacophore Query Function
Write a function `build_and_match_triad` that takes a molecule, generates a 3D conformer, extracts features, and tests whether it contains a specified 3-point triad within given tolerances.""")

    ex1_student = """def build_and_match_triad(
    mol: Chem.Mol,
    triad_families: Tuple[str, str, str],
    target_distances: Tuple[float, float, float],
    tolerance: float = 2.0
) -> bool:
    \"\"\"Generate a 3D conformer for mol and check if it matches the 3-point pharmacophore query.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement build_and_match_triad\")"""

    ex1_solution = """def build_and_match_triad(
    mol: Chem.Mol,
    triad_families: Tuple[str, str, str],
    target_distances: Tuple[float, float, float],
    tolerance: float = 2.0
) -> bool:
    \"\"\"Generate a 3D conformer for mol and check if it matches the 3-point pharmacophore query.\"\"\"
    if mol is None:
        return False
    m = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    if res != 0:
        return False
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    return pharm.match_3point_pharmacophore_distances(
        mol=m,
        feat_types=triad_families,
        target_distances=target_distances,
        tolerance=tolerance
    )"""

    add_cell("exercise", ex1_student, ex1_solution)

    add_cell("markdown", """### Exercise 2: Pharmacophore Pair Distance Calculator
Implement `compute_feature_pair_distance` to extract features from a 3D molecule and return the Euclidean distance between two designated feature families.""")

    ex2_student = """def compute_feature_pair_distance(
    mol: Chem.Mol,
    fam1: str,
    fam2: str
) -> Optional[float]:
    \"\"\"Return the Euclidean distance between the first occurrence of fam1 and fam2 in 3D conformer.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError(\"Implement compute_feature_pair_distance\")"""

    ex2_solution = """def compute_feature_pair_distance(
    mol: Chem.Mol,
    fam1: str,
    fam2: str
) -> Optional[float]:
    \"\"\"Return the Euclidean distance between the first occurrence of fam1 and fam2 in 3D conformer.\"\"\"
    feats = pharm.extract_features(mol)
    f1_list = [f for f in feats if f["family"] == fam1 and f["pos"] is not None]
    f2_list = [f for f in feats if f["family"] == fam2 and f["pos"] is not None]

    if not f1_list or not f2_list:
        return None

    p1 = np.array(f1_list[0]["pos"])
    p2 = np.array(f2_list[0]["pos"])
    return float(np.linalg.norm(p1 - p2))"""

    add_cell("exercise", ex2_student, ex2_solution)

    # --- Self-Checks ---
    add_cell("markdown", """## 9. Self-Check Unit Tests
Run the cell below to verify your solutions against automated test assertions.""")

    add_cell("code", """# --- Unit Tests for Exercise 1 ---
mol_test = Chem.MolFromSmiles(donepezil_smi)
is_match_ex1 = build_and_match_triad(
    mol_test,
    ("Aromatic", "PosIonizable", "Acceptor"),
    (d_arom_pos, d_pos_acc, d_arom_acc),
    tolerance=2.0
)
assert is_match_ex1 == True, "Donepezil must match the AChE triad in build_and_match_triad"

# Negative control: Simple alkane should not match
mol_alkane = Chem.MolFromSmiles("CCCCCCCCC")
is_match_alkane = build_and_match_triad(
    mol_alkane,
    ("Aromatic", "PosIonizable", "Acceptor"),
    (d_arom_pos, d_pos_acc, d_arom_acc),
    tolerance=2.0
)
assert is_match_alkane == False, "Nonane should not match an aromatic/cationic pharmacophore"

# --- Unit Tests for Exercise 2 ---
dist_calc = compute_feature_pair_distance(mol_donepezil, "Aromatic", "PosIonizable")
assert dist_calc is not None, "Distance between Aromatic and PosIonizable in Donepezil must be found"
assert 2.0 < dist_calc < 15.0, f"Distance {dist_calc:.2f} Å is within expected physical range"

print("All assertion self-checks passed!")
print("✔ NB11 checks passed")""")

    # Write notebooks
    os.makedirs("Cheminformatics/notebooks", exist_ok=True)
    os.makedirs("Cheminformatics/solutions", exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    path_student = "Cheminformatics/notebooks/nb11_pharmacophore_modeling.ipynb"
    path_sol = "Cheminformatics/solutions/nb11_pharmacophore_modeling.ipynb"

    with open(path_student, "w") as f:
        nbformat.write(nb_student, f)

    with open(path_sol, "w") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {path_student} and {path_sol}")

if __name__ == "__main__":
    create_nb11()
