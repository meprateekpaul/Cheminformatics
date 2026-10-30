import os
import sys
import json
from nb_helper import create_notebook, add_md, add_code, REPO_ROOT

def build_a1_starter():
    nb = create_notebook()

    add_md(nb, """# Assignment A1: Searching Chemical Libraries with a Template or Pharmacophore

**Course:** Practical Cheminformatics & Computer-Aided Drug Design  
**Total Points:** 15 Points (out of 30 for course assignments)  
**Deliverable:** Completed Jupyter notebook with code, outputs, and scientific write-up.

---

## 🎯 Objectives
1. **Curate Actives & Decoys:** Ingest known AChE inhibitors and property-matched decoys (`d04_chembl_ache_curated.csv` and `d17_ache_decoys.csv`).
2. **SMARTS Substructure Filter:** Define and evaluate a SMARTS query representing the core active pharmacophoric scaffold (e.g. basic benzyl-piperidine motif).
3. **3-Point Pharmacophore Model:** Formulate a geometric pharmacophore feature model using `chemlib.pharm`.
4. **Fingerprint Similarity Screening:** Perform ligand-based virtual screening using ECFP4 fingerprints with Tanimoto, Dice, and Cosine metrics.
5. **Screening Performance Benchmark:** Construct ROC and PR curves; compute **Enrichment Factor ($EF_{1\%}$)** and **BEDROC ($\alpha = 20$)**.
6. **Scaffold Diversity Selection:** Cluster the top 50 ranked hits using Bemis-Murcko scaffolds and select 10 diverse hits for purchasing recommendation.
""")

    add_md(nb, """## 1. Setup & Environment""")

    add_code(nb, """import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skmetrics

# Include chemlib
sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../src"))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import chemlib
from chemlib import fingerprints, metrics, pharm

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
""")

    add_md(nb, """## 2. Task 1: Ingestion & Dataset Curation (20% of grade)

Load the known actives (`data/processed/d04_chembl_ache_curated.csv`) and property-matched decoys (`data/processed/d17_ache_decoys.csv`).
- Define active molecules ($y=1$) as those with experimental $IC_{50} \le 1000\ \text{nM}$.
- Define decoys ($y=0$) as inactive molecules ($IC_{50} > 10,000\ \text{nM}$).
- Combine into a single unified screening library dataframe `df_library`.
""")

    add_code(nb, """# Load datasets
actives_path = os.path.join("..", "..", "data", "processed", "d04_chembl_ache_curated.csv")
decoys_path = os.path.join("..", "..", "data", "processed", "d17_ache_decoys.csv")

df_actives_raw = pd.read_csv(actives_path)
df_decoys_raw = pd.read_csv(decoys_path)

# Task 1: Assemble unified library
# YOUR CODE HERE
# Create df_library with columns: 'molecule_id', 'smiles', 'standard_value_nm', 'label'
raise NotImplementedError("Student Task 1: Assemble df_library")
""")

    add_code(nb, """# Verification check for Task 1
assert "df_library" in locals(), "df_library must be defined"
assert "label" in df_library.columns, "df_library must have a 'label' column"
assert df_library["label"].nunique() == 2, "df_library must contain both 0 (decoys) and 1 (actives)"
print(f"Task 1 Passed: Unified screening library contains {len(df_library)} molecules ({df_library['label'].sum()} actives).")
""")

    add_md(nb, """## 3. Task 2: SMARTS Substructure / Template Search (25% of grade)

1. Formulate a SMARTS pattern representing the core pharmacophore of donepezil-like AChE inhibitors (e.g. a benzyl-piperidine core `c1ccccc1CN2CCC` or an aromatic ring connected to a basic nitrogen).
2. Screen `df_library` using `mol.HasSubstructMatch(query_mol)`.
3. Calculate:
   - Hit rate ($\%$)
   - Precision: $TP / (TP + FP)$
   - Recall (Sensitivity): $TP / (TP + FN)$
""")

    add_code(nb, """# Task 2: Substructure Query
# Define your core SMARTS pattern
smarts_pattern = "c1ccccc1C[NH0,NH1]1CCCCC1"  # Or your chosen motif

# Filter the library and compute TP, FP, TN, FN
# YOUR CODE HERE
raise NotImplementedError("Student Task 2: Substructure filtering")
""")

    add_md(nb, """## 4. Task 3: 3-Point Pharmacophore Modeling (20% of grade)

Using `chemlib.pharm.extract_features` or RDKit feature factories:
1. Extract chemical features (HB Donor, HB Acceptor, Aromatic Ring, Hydrophobic) from reference active **Donepezil**.
2. Identify a 3-point pharmacophore triangle (e.g. Aromatic centroid 1, Aromatic centroid 2, and Basic/H-bond feature).
3. Compute inter-feature Euclidean distances (in Å) between the 3 points.
""")

    add_code(nb, """# Task 3: Pharmacophore extraction
ref_donepezil = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
mol_ref = Chem.MolFromSmiles(ref_donepezil)
mol_ref = Chem.AddHs(mol_ref)
AllChem.EmbedMolecule(mol_ref, randomSeed=42)

# Extract features and define 3-point pharmacophore triangle
# YOUR CODE HERE
raise NotImplementedError("Student Task 3: Pharmacophore triangle")
""")

    add_md(nb, """## 5. Task 4: Fingerprint Similarity Search & Evaluation (20% of grade)

1. Compute 2048-bit ECFP4 fingerprints for all molecules in `df_library`.
2. Compute **Tanimoto similarity** against the reference active (Donepezil).
3. Generate ROC and PR curves.
4. Calculate:
   - **ROC-AUC**
   - **$EF_{1\%}$** (Enrichment Factor at 1%)
   - **BEDROC ($\alpha = 20.0$)**
""")

    add_code(nb, """# Task 4: Fingerprint similarity screening
# YOUR CODE HERE
# Calculate tanimoto similarities, ROC-AUC, EF1%, and BEDROC
raise NotImplementedError("Student Task 4: Similarity search and metrics")
""")

    add_md(nb, """## 6. Task 5: Diversity Selection & Purchase Recommendation (15% of grade)

1. Extract the top 50 ranked hits from the virtual screen.
2. Group the top hits by **Bemis-Murcko scaffold**.
3. Select **10 structurally diverse candidate molecules** representing distinct scaffolds.
4. Provide a brief scientific justification for your final selection.
""")

    add_code(nb, """# Task 5: Scaffold diversity selection
# YOUR CODE HERE
# Select top 10 diverse hits
raise NotImplementedError("Student Task 5: Diversity clustering and selection")
""")

    add_code(nb, """print("Assignment A1 Notebook Loaded Successfully!")
""")

    a1_path = os.path.join(REPO_ROOT, "assignments", "A1_library_search", "starter.ipynb")
    os.makedirs(os.path.dirname(a1_path), exist_ok=True)
    with open(a1_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved Assignment A1 starter: {a1_path}")


def build_a2_starter():
    nb = create_notebook()

    add_md(nb, """# Assignment A2: Structure-Based De Novo Design and Fragment-Based Discovery

**Course:** Practical Cheminformatics & Computer-Aided Drug Design  
**Total Points:** 15 Points (out of 30 for course assignments)  
**Deliverable:** Completed Jupyter notebook with code, designed candidates, docked poses, and scientific report.

---

## 🎯 Objectives
1. **Binding Pocket Characterization:** Analyze the AChE active site gorge (catalytic triad Ser200, Glu327, His440 and aromatic gorge Trp84, Phe330) from `d10_1eve_ache_pocket.pdb`.
2. **Rule of Three Fragment Assembly:** Assemble a pool of fragment starting points adhering to Congreve's Rule of Three ($\text{MW} \le 300$, $c\log P \le 3$, $\text{HBD} \le 3$, $\text{HBA} \le 3$, $\text{RotB} \le 3$, $\text{TPSA} \le 60\ \text{Å}^2$).
3. **De Novo Generation via BRICS:** Apply fragment growing and linking rules to generate **$\ge 20$ complete de novo candidates**.
4. **Medicinal Chemistry Triage:** Eliminate PAINS motifs and ensure high synthetic accessibility ($\text{SA Score} \le 4.5$).
5. **AutoDock Vina Docking:** Dock candidates into the AChE active site pocket using `chemlib.docking`.
6. **Multi-Parameter Lead Prioritization:** Compute **Ligand Efficiency (LE)** and **Lipophilic Efficiency (LLE)** to prioritize top 5 drug candidates.
""")

    add_md(nb, """## 1. Setup & Environment""")

    add_code(nb, """import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Include chemlib
sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../src"))

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, BRICS
import chemlib
from chemlib import docking, descriptors

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
""")

    add_md(nb, """## 2. Task 1: Binding Pocket Characterization (20% of grade)

Load the prepared AChE pocket `data/processed/d10_1eve_ache_pocket.pdb`:
1. Extract catalytic triad residues (Ser200, Glu327, His440) and aromatic sub-pocket residues (Trp84, Phe330).
2. Calculate the pocket centroid $\mathbf{c}$ and bounding box dimensions with padding $8.0\ \text{Å}$.
""")

    add_code(nb, """pocket_path = os.path.join("..", "..", "data", "processed", "d10_1eve_ache_pocket.pdb")

# Task 1: Compute pocket center and dimensions
# YOUR CODE HERE
raise NotImplementedError("Student Task 1: Pocket characterization")
""")

    add_md(nb, """## 3. Task 2: Fragment Library Assembly & Rule of Three (20% of grade)

1. Select a set of starting fragment cores (e.g., indanone, piperidine, benzylamine, 7-azaindole, tacrine fragment).
2. Filter the fragments using Congreve's **Rule of Three**:
   - $\text{MW} \le 300\ \text{Da}$
   - $c\log P \le 3.0$
   - $\text{HBD} \le 3$
   - $\text{HBA} \le 3$
   - $\text{Rotatable Bonds} \le 3$
   - $\text{TPSA} \le 60\ \text{Å}^2$
""")

    add_code(nb, """# Task 2: Assemble and filter fragment starting points
# YOUR CODE HERE
raise NotImplementedError("Student Task 2: Rule of Three filtering")
""")

    add_md(nb, """## 4. Task 3: De Novo Candidate Assembly via BRICS (20% of grade)

Using BRICS reaction rules or fragment growing/linking:
1. Systematically attach substituents to the starting core fragment.
2. Generate **at least 20 novel, complete de novo lead molecules**.
3. Verify that all molecules are chemically valid and sanitized.
""")

    add_code(nb, """# Task 3: Generate >= 20 de novo molecules
# YOUR CODE HERE
raise NotImplementedError("Student Task 3: BRICS de novo generation")
""")

    add_md(nb, """## 5. Task 4: Medicinal Chemistry Triage (PAINS & SA Score) (20% of grade)

1. Screen candidate molecules against PAINS (Pan-Assay Interference Compounds) filters.
2. Filter for drug-likeness (Lipinski Rule of 5) and synthetic accessibility.
3. Retain only candidates suitable for progression into docking.
""")

    add_code(nb, """# Task 4: PAINS and drug-likeness filtering
# YOUR CODE HERE
raise NotImplementedError("Student Task 4: Medicinal chemistry filtering")
""")

    add_md(nb, """## 6. Task 5: Molecular Docking & Efficiency Ranking (LE / LLE) (20% of grade)

1. Generate 3D conformers (ETKDG v3 + MMFF94) for the filtered de novo candidates.
2. Convert to PDBQT and dock into the receptor pocket using `chemlib.docking.run_vina_docking`.
3. Extract predicted binding free energies ($\Delta G_{\text{dock}}$).
4. Calculate:
   - **Ligand Efficiency (LE):** $LE = -\Delta G / N_{\text{heavy}}$
   - **Lipophilic Efficiency (LLE):** $LLE = \text{pIC}_{50}^{\text{pred}} - c\log P$
5. Rank and prioritize the **top 5 de novo candidates** with interaction analysis.
""")

    add_code(nb, """# Task 5: Docking and Multi-Parameter Optimization (LE/LLE)
# YOUR CODE HERE
raise NotImplementedError("Student Task 5: Docking and prioritization")
""")

    add_code(nb, """print("Assignment A2 Notebook Loaded Successfully!")
""")

    a2_path = os.path.join(REPO_ROOT, "assignments", "A2_denovo_design", "starter.ipynb")
    os.makedirs(os.path.dirname(a2_path), exist_ok=True)
    with open(a2_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved Assignment A2 starter: {a2_path}")


if __name__ == "__main__":
    build_a1_starter()
    build_a2_starter()
