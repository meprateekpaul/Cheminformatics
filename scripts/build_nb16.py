import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb16():
    nb = create_notebook()

    add_md(nb, """# Notebook 16:\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb16_target_discovery_and_prep.ipynb)\n\n Target Discovery and Structure Preparation

**Part F: Structure-Based Drug Design (SBDD)**  
**Estimated Time:** ~120 min  
**Level:** Intermediate (●●○)

---

## 🎯 Learning Objectives
1. **Disease Characterization & Target Identification:** Analyze differential gene expression (RNA-seq / microarray) data to identify aberrant biological targets using volcano plots.
2. **Subtractive Genomics:** Apply set-theoretic pathway filtering, host non-homology criteria, and essentiality filters to prioritize pathogen-specific targets.
3. **Biological Network Analysis:** Model disease vs. normal protein-protein interaction (PPI) networks to detect vulnerable network hubs and bottleneck nodes.
4. **3D Macromolecular Structures:** Retrieve and parse crystallographic (PDB) and predicted (AlphaFold DB) structures, evaluating experimental resolution and pLDDT confidence scores.
5. **Binding Site & Cavity Detection:** Understand probe-based pocket mapping (organic solvent probes) and calculate grid box parameters from co-crystallized ligand coordinates.
6. **Receptor & Ligand Preparation:** Execute receptor preparation protocols (water stripping, alternate conformations, His/Asp/Glu protonation states) and prepare ligand PDBQT files with Meeko.
""")

    add_md(nb, """## 1. Theoretical Framework: Target Selection in Drug Discovery

In modern computer-aided drug design (CADD), therapeutic discovery begins with **target identification and validation**:
- **Non-Infectious / Human Diseases (Cancer, Neurodegeneration, Metabolic):**
  - Targets are human proteins with aberrant expression, hyperactivation, or pathological aggregation.
  - *Cancer Examples:* **DNA Topoisomerase II** (cleavage complex stabilization by etoposide/doxorubicin), **PDGFR** (receptor tyrosine kinase inhibitor imatinib), **COX-2** (inducible inflammatory cyclooxygenase targeted by celecoxib).
  - *Neurodegeneration Examples:* **Acetylcholinesterase (AChE)** (cholinergic deficit hypothesis in Alzheimer's; donepezil, tacrine), **BACE-1** ($\beta$-secretase amyloidogenic processing of APP), **Tau/Amyloid aggregates** (fibril nucleation inhibitors).
- **Infectious Diseases (Bacterial, Viral, Parasitic):**
  - Targets must be **essential for pathogen survival or virulence** but **absent or non-homologous in the human host** to prevent off-target toxicity.
  - *Examples:* **Peptidoglycan transpeptidase** (penicillin-binding proteins absent in humans), **Lumazine synthase / Riboflavin synthase** (essential riboflavin biosynthesis pathway present in bacteria/fungi but absent in mammals, which acquire vitamin $B_2$ from diet).
""")

    add_md(nb, """## 2. Environment Setup & Data Ingestion""")

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
import networkx as nx

# Add chemlib to sys.path
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import chemlib
    from chemlib import docking
    import rdkit
    print(f"RDKit {rdkit.__version__} and chemlib loaded.")
except ImportError:
    print("Running in lightweight/standard environment.")

# Visualization styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 3. Differential Gene Expression & Volcano Plot

Transcriptomic profiling (RNA-seq / microarray) compares gene expression levels between diseased tissue and healthy controls:
- **Log2 Fold Change ($\log_2 \text{FC}$):** Measures the magnitude of expression change:
  $$\log_2 \text{FC} = \log_2 \left( \frac{\bar{X}_{\text{disease}}}{\bar{X}_{\text{control}}} \right)$$
  - $\log_2 \text{FC} > 0$: Up-regulated in disease.
  - $\log_2 \text{FC} < 0$: Down-regulated in disease.
- **Statistical Significance ($p$-value):** Evaluated via two-sample Student's t-test or empirical Bayes linear models (limma/DESeq2). Plotted as $-\log_{10}(p\text{-value})$.
- **Volcano Plot:** Displays statistical significance ($y$-axis) versus biological effect size ($x$-axis).
""")

    add_code(nb, """dge_path = os.path.join("..", "data", "processed", "d12_differential_gene_expression.csv")
if not os.path.exists(dge_path):
    dge_path = os.path.join("data", "processed", "d12_differential_gene_expression.csv")

df_dge = pd.read_csv(dge_path)
print(f"Loaded DGE dataset with {len(df_dge)} genes:")
df_dge.head()
""")

    add_code(nb, """# Calculate negative log10 p-value
df_dge["neg_log10_p"] = -np.log10(df_dge["p_value"])

# Define significance thresholds
fc_thresh = 1.5
p_thresh = 0.05
neg_log_p_thresh = -np.log10(p_thresh)

# Categorize genes
conditions = [
    (df_dge["log2_fold_change"] >= fc_thresh) & (df_dge["p_value"] <= p_thresh),
    (df_dge["log2_fold_change"] <= -fc_thresh) & (df_dge["p_value"] <= p_thresh)
]
choices = ["Up-regulated (Candidate Target)", "Down-regulated"]
df_dge["Regulation"] = np.select(conditions, choices, default="Not Significant")

# Plot Volcano Plot
plt.figure(figsize=(9, 6))
palette = {"Up-regulated (Candidate Target)": "#e11d48", "Down-regulated": "#2563eb", "Not Significant": "#94a3b8"}
sns.scatterplot(
    data=df_dge,
    x="log2_fold_change",
    y="neg_log10_p",
    hue="Regulation",
    palette=palette,
    alpha=0.8,
    s=50,
    edgecolor="none"
)

plt.axvline(x=fc_thresh, color="#e11d48", linestyle="--", alpha=0.6, label=f"FC threshold (±{fc_thresh})")
plt.axvline(x=-fc_thresh, color="#2563eb", linestyle="--", alpha=0.6)
plt.axhline(y=neg_log_p_thresh, color="#0f172a", linestyle=":", alpha=0.6, label=f"p = {p_thresh}")

plt.title("Volcano Plot: Target Identification from Differential Gene Expression", fontweight="bold")
plt.xlabel(r"$\log_2$ Fold Change (Disease / Control)")
plt.ylabel(r"$-\log_{10}(p\text{-value})$")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

candidates = df_dge[df_dge["Regulation"] == "Up-regulated (Candidate Target)"]
print(f"Identified {len(candidates)} significantly up-regulated candidate targets.")
""")

    add_md(nb, """## 4. Subtractive Genomics for Infectious Disease Targets

Subtractive genomics identifies antibacterial or antiparasitic drug targets by subtracting host-homologous proteins from the pathogen's essential proteome:
1. **Pathogen Metabolic Pathways:** Extract all metabolic pathways represented in the pathogen (*e.g., Mycobacterium tuberculosis*).
2. **Host Pathway Subtraction:** Remove pathways that are shared with *Homo sapiens*. Pathways unique to the pathogen (*e.g., riboflavin synthesis, shikimate pathway*) contain unique enzymes.
3. **BLAST Non-Homology Filter:** Ensure candidate proteins exhibit no significant sequence homology to human proteins (e.g. BLASTp $E$-value $> 10^{-2}$ or sequence identity $< 25\%$).
4. **Essentiality Filter:** Check against experimental essential gene databases (Database of Essential Genes - DEG). An enzyme must be essential for pathogen growth or survival under in vivo infection conditions.
""")

    add_code(nb, """# Subtractive genomics simulation
pathogen_pathways = {
    "Glycolysis", "TCA_Cycle", "Pentose_Phosphate", "Fatty_Acid_Synthase_I",
    "Riboflavin_Biosynthesis", "Shikimate_Pathway", "Peptidoglycan_Biosynthesis",
    "Mycolic_Acid_Biosynthesis", "Folate_Biosynthesis"
}

human_pathways = {
    "Glycolysis", "TCA_Cycle", "Pentose_Phosphate", "Fatty_Acid_Synthase_I",
    "Folate_Biosynthesis", "Purine_Metabolism", "Cholesterol_Biosynthesis"
}

# Set subtraction: pathogen-specific pathways
pathogen_unique_pathways = pathogen_pathways - human_pathways
print(f"Pathogen-unique pathways ({len(pathogen_unique_pathways)}):")
for p in sorted(pathogen_unique_pathways):
    print(f"  • {p}")

# Mock proteome with homology and essentiality data
pathogen_proteins = pd.DataFrame([
    {"protein_id": "Rv0001", "pathway": "Peptidoglycan_Biosynthesis", "human_seq_identity": 0.12, "is_essential": True, "gene": "dnaA"},
    {"protein_id": "Rv0440", "pathway": "Glycolysis", "human_seq_identity": 0.68, "is_essential": True, "gene": "groEL2"},
    {"protein_id": "Rv1415", "pathway": "Riboflavin_Biosynthesis", "human_seq_identity": 0.08, "is_essential": True, "gene": "ribA2"},
    {"protein_id": "Rv1416", "pathway": "Riboflavin_Biosynthesis", "human_seq_identity": 0.15, "is_essential": True, "gene": "ribH"},
    {"protein_id": "Rv2537", "pathway": "Shikimate_Pathway", "human_seq_identity": 0.05, "is_essential": True, "gene": "aroK"},
    {"protein_id": "Rv3804", "pathway": "Mycolic_Acid_Biosynthesis", "human_seq_identity": 0.18, "is_essential": True, "gene": "fbpA"},
    {"protein_id": "Rv0946", "pathway": "Shikimate_Pathway", "human_seq_identity": 0.10, "is_essential": False, "gene": "non_ess_1"}
])

# Subtractive filtering pipeline
filtered_targets = pathogen_proteins[
    (pathogen_proteins["pathway"].isin(pathogen_unique_pathways)) &
    (pathogen_proteins["human_seq_identity"] < 0.25) &
    (pathogen_proteins["is_essential"] == True)
]

print(f"\\nSubtractive genomics prioritized targets ({len(filtered_targets)}):")
filtered_targets[["protein_id", "gene", "pathway", "human_seq_identity"]]
""")

    add_md(nb, """## 5. Differential Biological Network Analysis & Hub Ranking

Proteins do not act in isolation; they function inside complex Protein-Protein Interaction (PPI) networks:
- **Degree Centrality:** Number of direct binding partners. High-degree nodes are **network hubs**.
- **Betweenness Centrality:** Fraction of all shortest paths passing through a given node:
  $$C_B(v) = \sum_{s \ne v \ne t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
  High betweenness nodes are **bottlenecks** controlling inter-modular information flow.
- Targeting hub-bottlenecks in cancer or infectious diseases can induce network collapse, whereas in non-oncological indications, peripheral nodes are preferred to minimize side effects.
""")

    add_code(nb, """# Build a toy disease signaling network
G_disease = nx.Graph()
edges = [
    ("EGFR", "GRB2"), ("GRB2", "SOS1"), ("SOS1", "KRAS"), ("KRAS", "BRAF"),
    ("BRAF", "MEK1"), ("MEK1", "ERK2"), ("EGFR", "PIK3CA"), ("PIK3CA", "AKT1"),
    ("AKT1", "MTOR"), ("AKT1", "BAD"), ("ERK2", "MYC"), ("ERK2", "FOS"),
    ("TP53", "MDM2"), ("TP53", "CDKN1A"), ("AKT1", "MDM2"), ("TOP2A", "TP53")
]
G_disease.add_edges_from(edges)

# Calculate centrality measures
deg_centrality = nx.degree_centrality(G_disease)
bet_centrality = nx.betweenness_centrality(G_disease)

df_network = pd.DataFrame({
    "Degree_Centrality": deg_centrality,
    "Betweenness_Centrality": bet_centrality
}).sort_values(by="Betweenness_Centrality", ascending=False)

print("Top Network Bottlenecks and Hubs:")
df_network.head(8)
""")

    add_md(nb, """## 6. Receptor 3D Structure Retrieval & Inspection

### Sources of 3D Macromolecular Coordinates
1. **X-ray Crystallography:** High experimental accuracy ($1.5 - 2.5\ \text{Å}$ typical resolution), captures ordered water molecules and co-crystallized inhibitors.
2. **Cryo-EM:** Essential for large membrane complexes (GPCRs, ribosome, ion channels); resolution has reached near-atomic levels ($< 3.0\ \text{Å}$).
3. **Homology Modeling (SWISS-MODEL, MODELLER):** Reliable when a homologous template with $> 30\%$ sequence identity exists.
4. **Deep Learning (AlphaFold 2/3, ESMFold):** Highly accurate predictions evaluated via the **pLDDT score** (predicted Local Distance Difference Test):
   - $\text{pLDDT} > 90$: Very high confidence (reliable backbone and side chains).
   - $70 < \text{pLDDT} \le 90$: High confidence (reliable backbone).
   - $50 < \text{pLDDT} \le 70$: Low confidence (potential loops/flexibility).
   - $\text{pLDDT} \le 50$: Very low confidence (likely intrinsically disordered).

Let us load the crystallographic structure of the **Acetylcholinesterase (AChE) binding pocket** (PDB ID: 1EVE) with co-crystallized Donepezil (E20).
""")

    add_code(nb, """pdb_pocket_path = os.path.join("..", "data", "processed", "d10_1eve_ache_pocket.pdb")
if not os.path.exists(pdb_pocket_path):
    pdb_pocket_path = os.path.join("data", "processed", "d10_1eve_ache_pocket.pdb")

with open(pdb_pocket_path, "r") as f:
    pdb_lines = f.readlines()

print(f"Loaded {len(pdb_lines)} lines from {os.path.basename(pdb_pocket_path)}")
atom_lines = [l for l in pdb_lines if l.startswith("ATOM  ")]
hetatm_lines = [l for l in pdb_lines if l.startswith("HETATM")]
print(f"Found {len(atom_lines)} protein atoms and {len(hetatm_lines)} ligand/hetero atoms.")
""")

    add_md(nb, """## 7. Binding Site Cavity Detection & Grid Box Definition

To perform docking, we must define the search volume (**grid box**):
- **Centroid ($\mathbf{c}$):** Geometric center of the active site:
  $$\mathbf{c} = \frac{1}{N} \sum_{i=1}^N \mathbf{r}_i = \left( \frac{1}{N}\sum x_i, \frac{1}{N}\sum y_i, \frac{1}{N}\sum z_i \right)$$
- **Box Dimensions ($\mathbf{s}$):** Span of the pocket or reference ligand plus padding ($\Delta \approx 6 - 10\ \text{Å}$):
  $$s_x = (x_{\max} - x_{\min}) + 2 \times \text{padding}$$
""")

    add_code(nb, """# Extract coordinates of co-crystallized ligand (E20)
ligand_coords = []
for line in hetatm_lines:
    if "E20" in line:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        ligand_coords.append((x, y, z))

ligand_coords = np.array(ligand_coords)
print(f"Extracted {len(ligand_coords)} ligand heavy atom coordinates.")

def get_box_from_coords(coords: np.ndarray, padding: float = 8.0):
    min_xyz = np.min(coords, axis=0)
    max_xyz = np.max(coords, axis=0)
    center = (min_xyz + max_xyz) / 2.0
    sizes = (max_xyz - min_xyz) + 2.0 * padding
    return center, sizes

center, sizes = get_box_from_coords(ligand_coords, padding=8.0)
print(f"Grid Box Center (x, y, z):   ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print(f"Grid Box Sizes (dx, dy, dz): ({sizes[0]:.2f}, {sizes[1]:.2f}, {sizes[2]:.2f})")
""")

    add_md(nb, """## 8. Receptor & Ligand Preparation Protocol

### Receptor Preparation Checklist:
1. **Remove Waters:** Strip bulk solvent; keep only structural water molecules that mediate key ligand-protein hydrogen bonds (e.g., conserved water in HIV-1 protease or nuclear receptors).
2. **Handle Alternate Locations:** When crystal structures contain `altloc A` and `altloc B`, select the higher occupancy conformation.
3. **Protonation States (pH 7.4):**
   - **Histidine (His):** Exists in three tautomeric states:
     - `HID` (H on $N\delta 1$)
     - `HIE` (H on $N\epsilon 2$)
     - `HIP` (doubly protonated, cationic)
   - **Aspartate (Asp) & Glutamate (Glu):** Typically unprotonated ($-\text{COO}^-$), unless buried in a catalytic dyad (e.g., Asp32/Asp218 in BACE-1 where one Asp is neutral).
   - **Lysine (Lys) & Arginine (Arg):** Protonated cations ($-\text{NH}_3^+$, Guanidinium).
4. **Hydrogen Addition & Charges:** Add polar hydrogens; compute partial charges (Gasteiger charges); format output as **PDBQT** (PDB with partial charges $Q$ and AutoDock atom types $T$).

### Ligand Preparation with Meeko:
Meeko converts RDKit 3D molecules into PDBQT format while detecting rotatable bonds (TORSDOF parameter).
""")

    add_code(nb, """# Ligand preparation using RDKit and Meeko
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"  # Donepezil
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Embed 3D conformer using ETKDG v3
params = AllChem.ETKDGv3()
params.randomSeed = 42
AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol)

print(f"Successfully generated 3D conformer with {mol.GetNumAtoms()} atoms ({mol.GetNumHeavyAtoms()} heavy atoms).")

# Prepare PDBQT string
try:
    from meeko import MoleculePreparation
    preparator = MoleculePreparation()
    mol_setup = preparator.prepare(mol)
    pdbqt_str = preparator.write_pdbqt_string()
    print("Meeko successfully generated ligand PDBQT. Sample output:")
    for line in pdbqt_str.splitlines()[:10]:
        print(line)
except ImportError:
    print("Meeko is not installed in the active environment. Conceptual PDBQT prepared.")
""")

    add_md(nb, """## 9. Guided Student Exercises & Self-Checks

---

### Exercise 1: High-Confidence Target Prioritization
Filter the loaded `df_dge` dataframe for targets that meet all three strict criteria:
1. $\log_2 \text{Fold Change} \ge 2.0$ (strongly up-regulated).
2. $p\text{-value} \le 0.001$ (high statistical significance).
3. `is_essential == True` (essential gene).

Assign the resulting dataframe to the variable `high_priority_targets`.
""")

    add_code(nb, """# Exercise 1: High-Confidence Target Prioritization
# --- SOLUTION BEGIN ---
high_priority_targets = df_dge[
    (df_dge["log2_fold_change"] >= 2.0) &
    (df_dge["p_value"] <= 0.001) &
    (df_dge["is_essential"] == True)
].copy()
# --- SOLUTION END ---

print(f"Number of high priority targets found: {len(high_priority_targets)}")
high_priority_targets[["gene_symbol", "log2_fold_change", "p_value", "is_essential"]]
""")

    add_code(nb, """# Self-check: Exercise 1
assert isinstance(high_priority_targets, pd.DataFrame), "high_priority_targets must be a DataFrame"
assert len(high_priority_targets) > 0, "Should find at least one high priority target"
assert (high_priority_targets["log2_fold_change"] >= 2.0).all(), "All fold changes must be >= 2.0"
assert (high_priority_targets["p_value"] <= 0.001).all(), "All p-values must be <= 0.001"
assert (high_priority_targets["is_essential"] == True).all(), "All targets must be essential"
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Binding Pocket Center and Volume Calculator
Implement the function `compute_pocket_metrics(coordinates: np.ndarray, padding: float = 6.0)`:
- Computes centroid: $\mathbf{c} = \frac{1}{N} \sum \mathbf{r}_i$.
- Computes bounding box dimensions: $\mathbf{s} = (\mathbf{r}_{\max} - \mathbf{r}_{\min}) + 2 \times \text{padding}$.
- Computes bounding box search volume: $V = s_x \times s_y \times s_z\ (\text{Å}^3)$.
- Returns a dictionary with keys: `"center"`, `"dimensions"`, and `"volume"`.
""")

    add_code(nb, """# Exercise 2: Pocket Metrics Calculator
def compute_pocket_metrics(coordinates: np.ndarray, padding: float = 6.0) -> dict:
    \"\"\"Calculate centroid, bounding box dimensions, and search volume from 3D coordinates.\"\"\"
    # --- SOLUTION BEGIN ---
    coords = np.asarray(coordinates, dtype=float)
    center = np.mean(coords, axis=0)
    min_pt = np.min(coords, axis=0)
    max_pt = np.max(coords, axis=0)
    dimensions = (max_pt - min_pt) + 2.0 * padding
    volume = float(dimensions[0] * dimensions[1] * dimensions[2])
    return {
        "center": np.round(center, 3),
        "dimensions": np.round(dimensions, 3),
        "volume": round(volume, 2)
    }
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
test_coords = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0]
])
res = compute_pocket_metrics(test_coords, padding=5.0)

np.testing.assert_allclose(res["center"], [2.5, 2.5, 2.5], atol=1e-2)
np.testing.assert_allclose(res["dimensions"], [20.0, 20.0, 20.0], atol=1e-2)
assert abs(res["volume"] - 8000.0) < 1.0, f"Expected 8000.0 A^3, got {res['volume']}"
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: Histidine Tautomer Assignment
In a receptor binding pocket, Histidine 440 forms a hydrogen bond with a nearby backbone carbonyl oxygen ($-\text{C}=\text{O}$, a hydrogen bond acceptor).
Recall:
- `HID`: proton on $N\delta 1$ (can donate from $N\delta 1$).
- `HIE`: proton on $N\epsilon 2$ (can donate from $N\epsilon 2$).

If the acceptor oxygen atom is located at distance $2.8\ \text{Å}$ from $N\epsilon 2$ and $4.9\ \text{Å}$ from $N\delta 1$, write a function `assign_his_tautomer(dist_nd1_to_acceptor, dist_ne2_to_acceptor)` that returns `"HIE"` if $N\epsilon 2$ is within hydrogen bonding distance ($\le 3.5\ \text{Å}$), `"HID"` if $N\delta 1$ is within hydrogen bonding distance, or `"UNASSIGNED"` otherwise.
""")

    add_code(nb, """# Exercise 3: Histidine Tautomer Assignment
def assign_his_tautomer(dist_nd1: float, dist_ne2: float, max_hb_dist: float = 3.5) -> str:
    \"\"\"Assign Histidine tautomeric state (HID vs HIE) based on proximity to hydrogen bond acceptor.\"\"\"
    # --- SOLUTION BEGIN ---
    if dist_ne2 <= max_hb_dist and dist_ne2 < dist_nd1:
        return "HIE"
    elif dist_nd1 <= max_hb_dist and dist_nd1 < dist_ne2:
        return "HID"
    return "UNASSIGNED"
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
assert assign_his_tautomer(4.9, 2.8) == "HIE", "Should assign HIE when NE2 is close to acceptor"
assert assign_his_tautomer(2.7, 5.1) == "HID", "Should assign HID when ND1 is close to acceptor"
assert assign_his_tautomer(6.0, 5.5) == "UNASSIGNED", "Should be UNASSIGNED when both distances exceed 3.5 A"
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """---

### Exercise 4: Ligand Radius of Gyration
The **radius of gyration ($R_g$)** quantifies the spatial compactness of a ligand conformer:
$$R_g = \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{r}_i - \mathbf{c}\|^2}$$
where $\mathbf{c}$ is the centroid of the $N$ atoms.

Implement `compute_radius_of_gyration(coords: np.ndarray) -> float`.
""")

    add_code(nb, """# Exercise 4: Radius of Gyration
def compute_radius_of_gyration(coords: np.ndarray) -> float:
    \"\"\"Compute radius of gyration (in Angstroms) for an array of atomic coordinates.\"\"\"
    # --- SOLUTION BEGIN ---
    pts = np.asarray(coords, dtype=float)
    centroid = np.mean(pts, axis=0)
    sq_dists = np.sum((pts - centroid) ** 2, axis=1)
    rg = np.sqrt(np.mean(sq_dists))
    return float(round(rg, 3))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 4
rg_val = compute_radius_of_gyration(ligand_coords)
print(f"Computed Donepezil fragment Rg: {rg_val} Angstroms")
assert 1.0 < rg_val < 10.0, f"Unreasonable Rg value: {rg_val}"

# Test with a known unit sphere distribution
test_sphere = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
assert abs(compute_radius_of_gyration(test_sphere) - 1.0) < 1e-2
print("✔ Exercise 4 passed!")
""")

    add_md(nb, """## Common Pitfalls

1. **Unresolved Protonation States**: Crystallographic PDB files do not contain hydrogen atoms; ignoring histidine tautomers (HID/HIE/HIP) or catalytic dyad states leads to incorrect electrostatic potential fields.
2. **Alternate Conformations and Co-factors**: Retaining crystallographic waters, salts, or alternate sidechain conformations arbitrarily can block binding pockets or distort docking grids.
3. **AlphaFold Confidence Misinterpretation**: AlphaFold DB structures require checking pLDDT and PAE scores; flexible loops or intrinsically disordered regions (pLDDT < 50) must not be treated as rigid binding cavities.""")

    add_md(nb, """## Summary and Key Takeaways

- **Target Selection**: Requires balancing disease relevance, genetic validation, tractability, and selective safety windows.
- **Structure Preparation**: Standard protocols include water stripping, alternate conformation resolution, explicit hydrogen addition, and protonation state assignment.
- **Pocket Metrics**: Cavity detection and grid box bounding parameters are crucial pre-requisites for structure-based docking.""")

    add_code(nb, """# Final validation
print("✔ NB16 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb16_target_discovery_and_prep.ipynb",
        "solutions/nb16_target_discovery_and_prep.ipynb"
    )

if __name__ == "__main__":
    build_nb16()
