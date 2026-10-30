"""Build script for nb15: Druglikeness, Filtering and ADME-Tox Property Prediction."""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_nb15():
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
    add_cell("markdown", """# nb15 · Druglikeness, Filtering and ADME-Tox Property Prediction
**Cheminformatics Course · Part E: QSAR, ML and ADME**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb15_druglikeness_and_admet.ipynb)

### Objectives
1. Explain the historical development of **druglikeness rules** and distinguish between **druglike**, **leadlike**, and **fragmentlike** chemical spaces.
2. Implement and evaluate the classical property filter criteria:
   - **Lipinski's Rule of 5 (Ro5)** (MW $\\le$ 500, $\\log P \\le 5$, HBD $\\le 5$, HBA $\\le 10$).
   - **Veber's Rules** (Rotatable Bonds $\\le 10$, TPSA $\\le 140\\text{ \\AA}^2$).
   - **Ghose, Egan, and Muegge filters** for leadlikeness.
   - **Congreve's Rule of 3 (Ro3)** for fragment-based drug design (incorporating errata note E5: TPSA $\\le 60\\text{ \\AA}^2$).
3. Reproduce the **SwissADME BOILED-Egg model** (WLOGP vs TPSA) predicting passive gastrointestinal absorption (HIA) and blood-brain barrier (BBB) penetration.
4. Screen compound libraries for **structural alerts and PAINS** (Pan-Assay Interference Compounds) using RDKit's `FilterCatalog`.
5. Quantify synthetic accessibility and compound complexity via **SA Score**.
6. Construct an end-to-end multi-property filtering pipeline and visualize library attrition via waterfall plots.
""")

    # --- Setup & Imports ---
    add_cell("markdown", """## 1. Setup, Environment & Helper Functions""")

    add_cell("code", """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, FilterCatalog
from rdkit.Chem.Draw import MolsToGridImage

import chemlib
from chemlib import descriptors

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
    raise FileNotFoundError(f"Could not locate {filename} in project data/processed paths.")
""")

    # --- Section 2: Theory of Druglikeness ---
    add_cell("markdown", """## 2. Theoretical Foundations of Druglikeness

In 1997, Christopher Lipinski and coworkers published their seminal analysis of over 2,245 compounds that had reached Phase II clinical trials or higher in the World Drug Index (WDI). They observed that 90% of orally active commercial drugs satisfied four simple physicochemical cutoff rules:
- **Molecular Weight ($MW$)** $\\le 500\\text{ Da}$
- **Calculated Lipophilicity ($\\log P$)** $\\le 5.0$
- **Hydrogen Bond Donors (HBD)** $\\le 5$ (sum of OH and NH)
- **Hydrogen Bond Acceptors (HBA)** $\\le 10$ (sum of O and N atoms)

> **Important Conceptual Note (Slide 04/06):**
> Lipinski's Rule of 5 is a filter for **oral bioavailability and gastrointestinal absorption**, NOT for target binding potency! A molecule can bind with picomolar affinity to a kinase or GPCR yet fail completely as an oral drug if it violates Lipinski's rules. Furthermore, an allowed "one violation" tolerance is standard in modern screening.

| Rule System | Authors | MW (Da) | $\\log P$ | HBD | HBA | Rotatable Bonds | TPSA ($\\text{\\AA}^2$) | Target Domain |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Rule of 5** | Lipinski (1997) | $\\le 500$ | $\\le 5.0$ | $\\le 5$ | $\\le 10$ | — | — | Oral bioavailability |
| **Veber** | Veber (2002) | — | — | — | — | $\\le 10$ | $\\le 140$ | Bioavailability in rats |
| **Ghose** | Ghose (1999) | 160–480 | -0.4–5.6 | — | — | — | — | Comprehensive druglike |
| **Rule of 3 (Ro3)** | Congreve (2003) | $\\le 300$ | $\\le 3.0$ | $\\le 3$ | $\\le 3$ | $\\le 3$ | $\\le 60$ | Fragment-based screening |
""")

    # --- Section 3: Computing Physicochemical Properties ---
    add_cell("markdown", """## 3. Calculating Druglikeness Properties with RDKit & chemlib""")

    add_cell("code", """# Load FDA Approved Drugs dataset
fda_path = find_data_file("d03_fda_drugs.csv")
df_fda = pd.read_csv(fda_path)
smi_col = "smiles" if "smiles" in df_fda.columns else "canonical_smiles"
df_fda["mol"] = df_fda[smi_col].apply(Chem.MolFromSmiles)
df_fda = df_fda.dropna(subset=["mol"]).reset_index(drop=True)
print(f"Loaded {len(df_fda)} FDA approved drugs.")

# Compute Lipinski & Veber properties using chemlib.descriptors
records = []
for mol in df_fda["mol"]:
    props = descriptors.compute_lipinski_veber(mol)
    records.append(props)

df_props = pd.DataFrame(records)
df_merged = pd.concat([df_fda[["name", smi_col]], df_props], axis=1)
df_merged.head()
""")

    add_cell("code", """# Evaluate compliance rates among FDA approved drugs
ro5_pass_pct = df_merged["Passes_Lipinski"].mean() * 100.0
veber_pass_pct = df_merged["Passes_Veber"].mean() * 100.0
both_pass_pct = (df_merged["Passes_Lipinski"] & df_merged["Passes_Veber"]).mean() * 100.0

print(f"FDA Approved Drugs Compliance:")
print(f"  Lipinski Ro5 Pass Rate: {ro5_pass_pct:.1f}%")
print(f"  Veber Pass Rate:        {veber_pass_pct:.1f}%")
print(f"  Both Pass Rate:         {both_pass_pct:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_merged["MW"], kde=True, ax=axes[0], color="teal")
axes[0].axvline(500, color="crimson", linestyle="--", label="Ro5 MW cutoff (500 Da)")
axes[0].set_title("Molecular Weight Distribution")
axes[0].legend()

sns.scatterplot(data=df_merged, x="MolLogP", y="TPSA", hue="Passes_Lipinski", palette={True: "teal", False: "crimson"}, ax=axes[1])
axes[1].axvline(5.0, color="crimson", linestyle="--", label="LogP = 5")
axes[1].axhline(140.0, color="darkorange", linestyle="--", label="TPSA = 140 Å²")
axes[1].set_title("MolLogP vs TPSA")
axes[1].legend()
plt.tight_layout()
plt.show()
""")

    # --- Section 4: BOILED-Egg Model ---
    add_cell("markdown", """## 4. The SwissADME BOILED-Egg Model (HIA vs BBB)

The **BOILED-Egg** (Brain Or IntestinaL EstimateD permeation predictive model, Daina & Zoete 2016) is a graphical predictive model for:
1. **Passive gastrointestinal absorption (HIA)**: represented by the "White" of the egg.
2. **Blood-Brain Barrier (BBB) penetration**: represented by the "Yolk" of the egg.

The model plots **WLOGP** (Wildman-Crippen lipophilicity) on the y-axis vs **TPSA** (Topological Polar Surface Area) on the x-axis.
- High TPSA ($> 142\\text{ \\AA}^2$) prevents both intestinal absorption and BBB penetration.
- Moderate lipophilicity and low TPSA ($\\text{TPSA} < 79\\text{ \\AA}^2$, $0.4 < \\text{WLOGP} < 5.0$) enable blood-brain barrier permeability.
""")

    add_cell("code", """# Define geometric ellipse boundaries for BOILED-Egg
# Semi-axes and center coordinates derived from Daina & Zoete (ChemMedChem 2016)
def is_in_boiled_egg(tpsa: float, wlogp: float) -> Tuple[bool, bool]:
    \"\"\"Determine if a molecule falls in the HIA (white) and/or BBB (yolk) regions.\"\"\"
    # Ellipse equation: ((x - xc)*cos(theta) + (y - yc)*sin(theta))^2 / a^2 + (-(x - xc)*sin(theta) + (y - yc)*cos(theta))^2 / b^2 <= 1
    # Approximation of the published SwissADME boundaries:
    # HIA (White):
    dx_hia = tpsa - 71.0
    dy_hia = wlogp - 2.29
    in_hia = ((dx_hia / 71.0) ** 2 + (dy_hia / 3.1) ** 2) <= 1.0

    # BBB (Yolk):
    dx_bbb = tpsa - 38.0
    dy_bbb = wlogp - 2.5
    in_bbb = ((dx_bbb / 38.0) ** 2 + (dy_bbb / 1.8) ** 2) <= 1.0
    return in_hia, in_bbb

# Classify our FDA drugs
hia_flags, bbb_flags = [], []
for idx, row in df_merged.iterrows():
    hia, bbb = is_in_boiled_egg(row["TPSA"], row["MolLogP"])
    hia_flags.append(hia)
    bbb_flags.append(bbb)

df_merged["Predicted_HIA"] = hia_flags
df_merged["Predicted_BBB"] = bbb_flags
print(f"FDA Drugs in HIA (White): {df_merged['Predicted_HIA'].sum()} / {len(df_merged)} ({df_merged['Predicted_HIA'].mean()*100:.1f}%)")
print(f"FDA Drugs in BBB (Yolk):  {df_merged['Predicted_BBB'].sum()} / {len(df_merged)} ({df_merged['Predicted_BBB'].mean()*100:.1f}%)")
""")

    # --- Section 5: Structural Alerts & PAINS ---
    add_cell("markdown", """## 5. Structural Alerts: PAINS and Reactive Moieties

**Pan-Assay Interference Compounds (PAINS)** are chemical compounds that often yield false positives in high-throughput screening assays via non-specific mechanisms:
- Protein reactivity / covalent adduct formation
- Redox cycling producing reactive oxygen species (ROS)
- Photoreactivity or autofluorescence
- Micellar aggregation

RDKit provides the `FilterCatalog` module containing the full Baell & Holloway (2010) PAINS collections (Catalog A, B, C).
""")

    add_cell("code", """# Build RDKit PAINS FilterCatalog
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(params)

# Test on classic PAINS compounds
sample_smiles = [
    ("Rhodanine (PAINS)", "O=C1NC(=S)SC1=Cc1ccc(O)cc1"),
    ("Curcumin (PAINS)", "O=C(C=Cc1ccc(O)c(OC)c1)CC(=O)C=Cc2ccc(O)c(OC)c2"),
    ("Aspirin (Clean)", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Donepezil (Clean)", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"),
]

for name, smi in sample_smiles:
    m = Chem.MolFromSmiles(smi)
    has_pains = pains_catalog.HasMatch(m)
    entry = pains_catalog.GetFirstMatch(m)
    desc = entry.GetDescription() if entry else "None"
    print(f"{name:20s}: PAINS Alert = {str(has_pains):5s} | Detail: {desc}")
""")

    # --- Section 6: Exercises ---
    add_cell("markdown", """## 6. Exercises

### Exercise 1: Multi-Property Filter Implementation
Implement a function `filter_congreve_rule_of_three(mol)` that verifies whether a molecule satisfies the **Congreve Rule of 3 (Ro3)** for fragment libraries:
- $MW \\le 300\\text{ Da}$
- $\\text{MolLogP} \\le 3.0$
- $HBD \\le 3$
- $HBA \\le 3$
- $\\text{Rotatable Bonds} \\le 3$
- $TPSA \\le 60.0\\text{ \\AA}^2$ (incorporating Errata E5)
""")

    ex1_stub = """def filter_congreve_rule_of_three(mol: Chem.Mol) -> bool:
    \"\"\"Return True if molecule passes Congreve's Rule of 3, False otherwise.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError
"""
    ex1_sol = """def filter_congreve_rule_of_three(mol: Chem.Mol) -> bool:
    \"\"\"Return True if molecule passes Congreve's Rule of 3, False otherwise.\"\"\"
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    return (mw <= 300.0) and (logp <= 3.0) and (hbd <= 3) and (hba <= 3) and (rotb <= 3) and (tpsa <= 60.0)
"""
    add_cell("exercise", ex1_stub, ex1_sol)

    add_cell("code", """# Test Exercise 1
frag1 = Chem.MolFromSmiles("c1ccccc1O")  # Phenol: MW 94, logP 1.4, HBD 1, HBA 1, rotb 0, TPSA 20.2 -> PASS
frag2 = Chem.MolFromSmiles("COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2")  # Donepezil: MW 379 -> FAIL

assert filter_congreve_rule_of_three(frag1) is True
assert filter_congreve_rule_of_three(frag2) is False
print("✔ Exercise 1 passed!")
""")

    add_cell("markdown", """### Exercise 2: PAINS Screening Function
Implement a function `screen_library_pains(smiles_list)` that takes a list of SMILES strings, screens them against the RDKit PAINS catalog, and returns a list of boolean flags indicating if each compound has an alert.
""")

    ex2_stub = """def screen_library_pains(smiles_list: List[str]) -> List[bool]:
    \"\"\"Screen a list of SMILES for PAINS alerts. Returns list of bool (True if alert present).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError
"""
    ex2_sol = """def screen_library_pains(smiles_list: List[str]) -> List[bool]:
    \"\"\"Screen a list of SMILES for PAINS alerts. Returns list of bool (True if alert present).\"\"\"
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    results = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            results.append(catalog.HasMatch(m))
        else:
            results.append(False)
    return results
"""
    add_cell("exercise", ex2_stub, ex2_sol)

    add_cell("code", """# Test Exercise 2
test_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",                # Aspirin (False)
    "O=C1NC(=S)SC1=Cc1ccc(O)cc1",           # Rhodanine (True)
    "c1ccccc1N",                             # Aniline (False)
]
alerts = screen_library_pains(test_smiles)
assert alerts == [False, True, False]
print("✔ Exercise 2 passed!")
""")

    # --- Summary & Self-test ---
    add_cell("markdown", """## 7. Summary & Key Takeaways

1. **Druglikeness vs Affinity**: Druglikeness rules (Lipinski, Veber, Ghose) govern pharmacokinetics and oral absorption, not receptor binding potency.
2. **Rule of 5 Violations**: Many blockbuster oral drugs (e.g. Atorvastatin) have 1 violation. Zero violations is typical for leadlikeness; one violation is acceptable for druglikeness.
3. **The BOILED-Egg Model**: Plots WLOGP vs TPSA to provide an intuitive visualization of gastrointestinal absorption and blood-brain barrier permeability.
4. **PAINS Screening**: PAINS filters identify promiscuous assay interferers early in the discovery funnel before costly follow-up synthesis.
""")

    add_cell("code", """# Final self-check for nb15
assert df_merged is not None
assert len(df_merged) > 0
assert "Passes_Lipinski" in df_merged.columns
assert "Passes_Veber" in df_merged.columns
print("✔ NB15 checks passed")
""")

    nb_stud = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)
    return nb_stud, nb_sol

def main():
    stud_dir = "Cheminformatics/notebooks"
    sol_dir = "Cheminformatics/solutions"
    os.makedirs(stud_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    nb_stud, nb_sol = create_nb15()

    stud_path = os.path.join(stud_dir, "nb15_druglikeness_and_admet.ipynb")
    sol_path = os.path.join(sol_dir, "nb15_druglikeness_and_admet.ipynb")

    with open(stud_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_stud, f)
    with open(sol_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_sol, f)

    print(f"Successfully generated {stud_path} and {sol_path}")

if __name__ == "__main__":
    main()
