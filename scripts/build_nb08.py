import json, os, ast
from pathlib import Path

def create_nb08(is_solution=False):
    cells = []
    
    # 1. Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NB08 · Molecular Descriptors: From Topology to 3D Conformations\n",
            "\n",
            "**Part C: Representations & Descriptors** · Estimated time: ~120 min · Difficulty: ●●●\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb08_molecular_descriptors.ipynb)\n",
            "\n",
            "---\n",
            "\n",
            "### One-Line Summary\n",
            "Master the 0D–4D descriptor taxonomy, implement classical 2D topological indices (Zagreb, Randić, Wiener, MWC) from scratch in NumPy, correlate the Randić index with experimental alkane boiling points, calculate 3D geometrical shapes (PMI, NPR1/NPR2, SASA), compare major toolkits (RDKit, Mordred, PaDEL), and execute an automated descriptor cleaning pipeline."
        ]
    })

    # 2. Objectives
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Learning Objectives\n",
            "\n",
            "By the end of this notebook, you will be able to:\n",
            "1. **Categorize Descriptors (0D to 4D)**: Classify molecular descriptors according to Todeschini & Consonni's taxonomy and evaluate them against the 13 desirable QSAR attributes.\n",
            "2. **Implement Topological Graph Invariants**: Derive and compute vertex degree, total adjacency, Zagreb $M_1$ and $M_2$, Randić connectivity index ($\\chi$), Wiener index ($W$), and molecular walk counts ($MWC_k$) directly from adjacency and distance matrices.\n",
            "3. **Validate Structure-Property Correlations**: Model experimental alkane boiling points ($C_1$ to $C_8$) as a function of the Randić branching index using linear regression.\n",
            "4. **Compute 3D Geometrical Descriptors**: Generate 3D conformers to compute the radius of gyration ($R_g$), Principal Moments of Inertia ratios (NPR1, NPR2), and Solvent-Accessible Surface Area (SASA).\n",
            "5. **Examine 4D Conformer Ensembles**: Evaluate the variance of shape and surface descriptors across conformer ensembles generated via ETKDG.\n",
            "6. **Clarify Quantum-Chemical Descriptors**: Correctly define HOMO and LUMO (Lowest *Unoccupied* Molecular Orbital; addressing Errata E9) and understand chemical hardness and dipole moments.\n",
            "7. **Execute Descriptor Cleaning**: Filter near-constant columns, impute missing values, and eliminate collinear features ($|r| > 0.95$) using `clean_descriptor_matrix`."
        ]
    })

    # 3. Prerequisites
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Prerequisites & Lecture Links\n",
            "\n",
            "- **Prerequisites**: Molecular graphs and matrix representations (`nb04`), 3D coordinate generation (`nb05`), functional group SMARTS (`nb07`).\n",
            "- **Lecture Aligned**: Lecture Deck 05 (*Molecular Descriptors*) and Lecture Deck 03 (*Cheminformatics Tools*).\n",
            "- **Core Concepts**: Topological indices, graph invariants, Randić branching, Wiener distance sum, principal moments of inertia, NPR shape space, collinearity filtering."
        ]
    })

    # 4. Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup and environment configuration\n",
            "import os, sys, random, warnings, math\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import stats\n",
            "\n",
            "from rdkit import Chem, RDLogger\n",
            "from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors\n",
            "\n",
            "# Ensure src directory is in sys.path\n",
            "repo_root = Path.cwd().parent if Path.cwd().name in [\"notebooks\", \"solutions\"] else Path.cwd()\n",
            "if str(repo_root / \"src\") not in sys.path:\n",
            "    sys.path.insert(0, str(repo_root / \"src\"))\n",
            "\n",
            "import chemlib\n",
            "from chemlib import graphs, descriptors\n",
            "\n",
            "SEED = 42\n",
            "FULL_RUN = False\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
            "RDLogger.DisableLog(\"rdApp.*\")\n",
            "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "print(\"Environment initialized successfully.\")"
        ]
    })

    # 5. Taxonomy of Descriptors
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. The Molecular Descriptor Concept & Taxonomy\n",
            "\n",
            "> \"The molecular descriptor is the final result of a logic and mathematical procedure which transforms every chemical information encoded within a symbolic representation of a molecule into a useful number or the result of some standardized experiment.\" \n",
            "> — **Roberto Todeschini & Viviana Consonni**, *Handbook of Molecular Descriptors* (Wiley-VCH)\n",
            "\n",
            "### Dimensionality Hierarchy (0D to 4D)\n",
            "\n",
            "| Dimension | Input Representation | Examples | Sensitivity to Conformation |\n",
            "|:---|:---|:---|:---|\n",
            "| **0D** | Molecular Formula | Atom counts, Molecular Weight, Halogen count | Invariant |\n",
            "| **1D** | Substructure / Fingerprint list | Functional group counts, Fragment counts (Checkmol) | Invariant |\n",
            "| **2D** | Molecular Graph (Atoms & Bonds) | Zagreb $M_1/M_2$, Randić $\\chi$, Wiener $W$, TPSA, cLogP | Invariant |\n",
            "| **3D** | Single 3D Conformation | Radius of Gyration ($R_g$), PMI ratios (NPR1/2), SASA, Dipole | **High** (depends on conformer) |\n",
            "| **4D** | Dynamic Conformer Ensemble | Ensemble average SASA, Boltzmann-weighted field potentials | Averages over conformational space |\n",
            "\n",
            "### The 13 Desirable Attributes for Descriptors in QSAR\n",
            "From Lecture 05, robust QSAR models require descriptors that satisfy:\n",
            "1. Clear chemical and physical interpretation.\n",
            "2. Direct relation to free energy or binding mechanism.\n",
            "3. Invariance to molecular translation, rotation, and atom numbering.\n",
            "4. Well-defined mathematical properties and scaling.\n",
            "5. Efficient and reproducible algorithmic computation.\n",
            "6. Continuous or well-behaved discrete numerical domain.\n",
            "7. Low pairwise collinearity across the feature set.\n",
            "8. Minimal sensitivity to minor conformational noise (for 3D descriptors)."
        ]
    })

    # 6. 2D Topological Descriptors Implementation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. 2D Topological Descriptors: Mathematical Implementation\n",
            "\n",
            "Topological indices are numerical graph invariants that capture size, branching, and cyclicity from the hydrogen-depleted molecular graph $G = (V, E)$.\n",
            "\n",
            "### Core Formulas\n",
            "Let $A$ be the binary adjacency matrix, $D$ the topological distance matrix, and $\\delta_i$ the vertex degree (number of heavy neighbors) of atom $i$:\n",
            "\n",
            "1. **Total Adjacency Index ($A_{\\text{tot}}$)**:\n",
            "   $$A_{\\text{tot}} = \\sum_{i} \\sum_{j} A_{ij} = 2 \\times |E| = 2 \\times \\text{bonds}$$\n",
            "\n",
            "2. **First Zagreb Index ($M_1$)**:\n",
            "   $$M_1 = \\sum_{i \\in V} \\delta_i^2 = \\sum_{(i,j) \\in E} (\\delta_i + \\delta_j)$$\n",
            "\n",
            "3. **Second Zagreb Index ($M_2$)**:\n",
            "   $$M_2 = \\sum_{(i,j) \\in E} \\delta_i \\delta_j$$\n",
            "\n",
            "4. **Randić Branching / Connectivity Index ($\\chi$)**:\n",
            "   $$\\chi = \\sum_{(i,j) \\in E} \\frac{1}{\\sqrt{\\delta_i \\delta_j}}$$\n",
            "\n",
            "5. **Wiener Index ($W$)**:\n",
            "   $$W = \\frac{1}{2} \\sum_{i \\in V} \\sum_{j \\in V} D_{ij}$$\n",
            "\n",
            "6. **Molecular Walk Counts ($MWC_k$)**:\n",
            "   $$MWC_k = \\sum_{i} \\sum_{j} [A^k]_{ij} \\quad (MWC_0 = |V|, \\; MWC_1 = 2|E|)$$"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step-by-step verification on 2-Methylbutane (Isopentane)\n",
            "# SMILES: CCC(C)C\n",
            "isopentane = Chem.MolFromSmiles(\"CCC(C)C\")\n",
            "\n",
            "A = graphs.get_adjacency_matrix(isopentane)\n",
            "D = graphs.get_distance_matrix(isopentane)\n",
            "\n",
            "print(\"Adjacency matrix A (heavy atoms):\")\n",
            "print(A)\n",
            "print(\"\\nDistance matrix D (shortest paths):\")\n",
            "print(D)\n",
            "\n",
            "# Compute indices via chemlib.graphs\n",
            "m1 = graphs.zagreb_m1_index(isopentane)\n",
            "m2 = graphs.zagreb_m2_index(isopentane)\n",
            "chi = graphs.randic_index(isopentane)\n",
            "w = graphs.wiener_index(isopentane)\n",
            "mwc0 = graphs.molecular_walk_count(isopentane, 0)\n",
            "mwc1 = graphs.molecular_walk_count(isopentane, 1)\n",
            "\n",
            "print(f\"\\n2-Methylbutane Graph Invariants:\")\n",
            "print(f\"  First Zagreb Index (M1): {m1}\")\n",
            "print(f\"  Second Zagreb Index (M2): {m2}\")\n",
            "print(f\"  Randić Connectivity Index (chi): {chi:.4f}\")\n",
            "print(f\"  Wiener Index (W): {w}\")\n",
            "print(f\"  MWC_0: {mwc0} (Heavy atoms: {isopentane.GetNumHeavyAtoms()})\")\n",
            "print(f\"  MWC_1: {mwc1} (2 * Bonds: {2 * isopentane.GetNumBonds()})\")"
        ]
    })

    # 7. Correlating Randic Index with Alkane Boiling Points
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Physical Validation: Randić Index vs. Alkane Boiling Points\n",
            "\n",
            "Milan Randić introduced the branching connectivity index $\\chi$ in 1975 (*J. Am. Chem. Soc.* **1975**, 97, 6609–6615) specifically to predict the physical properties of alkanes.\n",
            "\n",
            "Let's load the experimental alkane boiling point dataset (`data/processed/d02_alkanes_bp.csv`) covering $C_1$ to $C_8$ linear and branched alkanes, calculate $\\chi$ for each, and model the boiling point using linear regression."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load alkane boiling points dataset\n",
            "bp_file = repo_root / \"data\" / \"processed\" / \"d02_alkanes_bp.csv\"\n",
            "df_alkanes = pd.read_csv(bp_file)\n",
            "\n",
            "# Compute Randić index and Wiener index for each alkane\n",
            "randic_vals = []\n",
            "wiener_vals = []\n",
            "for smi in df_alkanes[\"smiles\"]:\n",
            "    m = Chem.MolFromSmiles(smi)\n",
            "    randic_vals.append(graphs.randic_index(m))\n",
            "    wiener_vals.append(graphs.wiener_index(m))\n",
            "\n",
            "df_alkanes[\"randic_index\"] = randic_vals\n",
            "df_alkanes[\"wiener_index\"] = wiener_vals\n",
            "\n",
            "# Linear regression: Boiling Point (°C) vs Randić index\n",
            "slope, intercept, r_val, p_val, std_err = stats.linregress(df_alkanes[\"randic_index\"], df_alkanes[\"boiling_point_C\"])\n",
            "r2 = r_val ** 2\n",
            "\n",
            "print(f\"Linear Regression: Boiling Point = {slope:.2f} * Randić + {intercept:.2f}\")\n",
            "print(f\"Pearson correlation r = {r_val:.4f}\")\n",
            "print(f\"Coefficient of determination R² = {r2:.4f}\")\n",
            "\n",
            "# Plot correlation\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "sns.regplot(data=df_alkanes, x=\"randic_index\", y=\"boiling_point_C\", ax=ax, \n",
            "            scatter_kws={\"s\": 60, \"color\": \"#1f77b4\"}, line_kws={\"color\": \"#d62728\", \"lw\": 2})\n",
            "ax.set_title(f\"Randić Connectivity Index vs Alkane Boiling Point (R² = {r2:.3f})\", fontsize=13, fontweight=\"bold\")\n",
            "ax.set_xlabel(\"Randić Connectivity Index (χ)\")\n",
            "ax.set_ylabel(\"Experimental Boiling Point (°C)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 8. 3D Geometrical Descriptors
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. 3D Geometrical Descriptors: Shape and Surface\n",
            "\n",
            "Unlike 2D graph invariants, 3D descriptors quantify the spatial geometry, volume, and shape of a specific molecular conformation.\n",
            "\n",
            "### 1. Radius of Gyration ($R_g$)\n",
            "Measures the radial distribution of heavy atom mass around the center of mass $\\mathbf{r}_{cm}$:\n",
            "$$R_g = \\sqrt{\\frac{1}{M} \\sum_{i} m_i \\|\\mathbf{r}_i - \\mathbf{r}_{cm}\\|^2}$$\n",
            "\n",
            "### 2. Principal Moments of Inertia & NPR Shape Space\n",
            "The inertia tensor has three principal moments $I_1 \\le I_2 \\le I_3$.\n",
            "Normalized Principal Moment Ratios (Sauer & Schwarz, 2003):\n",
            "$$\\text{NPR}_1 = \\frac{I_1}{I_3}, \\quad \\text{NPR}_2 = \\frac{I_2}{I_3}$$\n",
            "\n",
            "Molecules map onto a triangular shape space with vertices:\n",
            "- **$(0, 1)$ = Rod-like (Linear)**: $I_1 \\approx 0, I_2 = I_3$ (e.g., diacetylene).\n",
            "- **$(0.5, 0.5)$ = Disc-like (Planar)**: $I_1 = I_2 = 0.5 I_3$ (e.g., benzene, coronene).\n",
            "- **$(1, 1)$ = Spherical (Isotropic)**: $I_1 = I_2 = I_3$ (e.g., adamantane, methane)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demonstrate 3D shape descriptors for diverse geometry types\n",
            "shape_probes = [\n",
            "    (\"Benzene (Disc)\", \"c1ccccc1\"),\n",
            "    (\"Adamantane (Sphere)\", \"C1C2CC3CC1CC(C2)C3\"),\n",
            "    (\"Decane (Rod)\", \"CCCCCCCCCC\"),\n",
            "    (\"Aspirin\", \"CC(=O)Oc1ccccc1C(=O)O\"),\n",
            "]\n",
            "\n",
            "shape_data = []\n",
            "for name, smi in shape_probes:\n",
            "    m = Chem.MolFromSmiles(smi)\n",
            "    m3d = Chem.AddHs(m)\n",
            "    AllChem.EmbedMolecule(m3d, randomSeed=SEED)\n",
            "    AllChem.MMFFOptimizeMolecule(m3d)\n",
            "    \n",
            "    rg = rdMolDescriptors.CalcRadiusOfGyration(m3d)\n",
            "    npr1 = rdMolDescriptors.CalcNPR1(m3d)\n",
            "    npr2 = rdMolDescriptors.CalcNPR2(m3d)\n",
            "    tpsa = Descriptors.TPSA(m)\n",
            "    \n",
            "    shape_data.append({\n",
            "        \"Compound\": name,\n",
            "        \"Radius_of_Gyration\": round(rg, 3),\n",
            "        \"NPR1\": round(npr1, 3),\n",
            "        \"NPR2\": round(npr2, 3),\n",
            "        \"TPSA\": round(tpsa, 2)\n",
            "    })\n",
            "\n",
            "df_shape = pd.DataFrame(shape_data)\n",
            "df_shape"
        ]
    })

    # 9. 4D Descriptors & Conformer Ensembles
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. 4D Descriptors: Conformer Ensemble Averaging\n",
            "\n",
            "A flexible molecule in solution does not exist in a single static energy minimum, but rather as an ensemble of interconverting conformers. **4D descriptors** account for conformational entropy and thermal dynamics by averaging descriptors across the conformer population.\n",
            "\n",
            "Let's sample 30 diverse conformers for **Propranolol** (a flexible $\\beta$-blocker with 6 rotatable bonds) and observe the variation in Radius of Gyration."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate conformer ensemble for Propranolol\n",
            "propranolol_smi = \"CC(C)NCC(O)COc1cccc2ccccc12\"\n",
            "mol_prop = Chem.MolFromSmiles(propranolol_smi)\n",
            "mol_prop_3d = Chem.AddHs(mol_prop)\n",
            "\n",
            "# Generate 30 conformers using ETKDGv3\n",
            "params = AllChem.ETKDGv3()\n",
            "params.randomSeed = SEED\n",
            "conf_ids = list(AllChem.EmbedMultipleConfs(mol_prop_3d, numConfs=30, params=params))\n",
            "\n",
            "rg_ensemble = []\n",
            "for cid in conf_ids:\n",
            "    AllChem.MMFFOptimizeMolecule(mol_prop_3d, confId=cid)\n",
            "    rg = rdMolDescriptors.CalcRadiusOfGyration(mol_prop_3d, confId=cid)\n",
            "    rg_ensemble.append(rg)\n",
            "\n",
            "print(f\"Propranolol Conformer Ensemble (N={len(rg_ensemble)}):\")\n",
            "print(f\"  Mean Rg:   {np.mean(rg_ensemble):.3f} Å\")\n",
            "print(f\"  Std Dev:   {np.std(rg_ensemble):.3f} Å\")\n",
            "print(f\"  Min (compact): {np.min(rg_ensemble):.3f} Å\")\n",
            "print(f\"  Max (extended): {np.max(rg_ensemble):.3f} Å\")"
        ]
    })

    # 10. Quantum Chemical Descriptors (Errata E9)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Quantum-Chemical Descriptors & Errata Clarification\n",
            "\n",
            "> [!IMPORTANT]\n",
            "> **Lecture Errata (E9)**: In some introductory course slide decks, **LUMO** is erroneously described as \"Lowest *Occupied* Molecular Orbital\". **LUMO is the Lowest *Unoccupied* Molecular Orbital**, while **HOMO** is the Highest Occupied Molecular Orbital.\n",
            "\n",
            "Key Quantum Chemical Concepts:\n",
            "- **$E_{\\text{HOMO}}$**: Related to ionization potential ($IP \\approx -E_{\\text{HOMO}}$ via Koopmans' theorem). Indicates electron-donating nucleophilic capacity.\n",
            "- **$E_{\\text{LUMO}}$**: Related to electron affinity ($EA \\approx -E_{\\text{LUMO}}$). Indicates electron-accepting electrophilic capacity.\n",
            "- **HOMO–LUMO Energy Gap ($\\Delta E = E_{\\text{LUMO}} - E_{\\text{HOMO}}$)**: Measures kinetic stability and chemical reactivity (large gap = hard/unreactive molecule; narrow gap = soft/reactive polarizable molecule).\n",
            "- **Chemical Hardness ($\\eta$)**: $\\eta \\approx \\frac{IP - EA}{2} = \\frac{E_{\\text{LUMO}} - E_{\\text{HOMO}}}{2}$."
        ]
    })

    # 11. Tool Comparison
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Cheminformatics Descriptor Toolkits: RDKit vs. Mordred vs. PaDEL\n",
            "\n",
            "| Toolkit | Descriptors Count | Language / Backend | 3D Support | Primary Strength |\n",
            "|:---|:---|:---|:---|:---|\n",
            "| **RDKit** | ~ 210 standard 2D/3D | C++ / Python bindings | Yes (conformer required for 3D) | Standard, blazing fast, rock solid |\n",
            "| **Mordred** | > 1,600 descriptors | Pure Python / NetworkX | Yes (`ignore_3D=False`) | Enormous coverage, standard in modern QSAR |\n",
            "| **PaDEL-Descriptor** | 797 1D/2D + 134 3D | Java (via `padelpy`) | Yes | Historical benchmark; cited widely in legacy literature |"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute standard 2D RDKit descriptors for the drug panel\n",
            "drugs_subset = [\n",
            "    Chem.MolFromSmiles(\"CC(=O)Oc1ccccc1C(=O)O\"),     # Aspirin\n",
            "    Chem.MolFromSmiles(\"CC(=O)Nc1ccc(O)cc1\"),         # Paracetamol\n",
            "    Chem.MolFromSmiles(\"CC(C)Cc1ccc(C(C)C(=O)O)cc1\"), # Ibuprofen\n",
            "    Chem.MolFromSmiles(\"Cn1cnc2c1c(=O)n(C)c(=O)n2C\"), # Caffeine\n",
            "]\n",
            "\n",
            "df_rdkit_desc = descriptors.compute_rdkit_descriptors(drugs_subset)\n",
            "print(f\"Computed {df_rdkit_desc.shape[1]} RDKit descriptors for {len(drugs_subset)} compounds.\")\n",
            "print(\"Sample descriptors:\", list(df_rdkit_desc.columns[:6]))\n",
            "df_rdkit_desc[list(df_rdkit_desc.columns[:6])]"
        ]
    })

    # 12. Descriptor Cleaning Pipeline
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. The Descriptor Matrix Cleaning Pipeline\n",
            "\n",
            "Computing large descriptor suites (e.g. from Mordred or RDKit) generates raw feature matrices plagued by three major pathologies:\n",
            "1. **Missing values (NaNs)**: Failed 3D conformers or divide-by-zero errors in topological indices.\n",
            "2. **Zero or near-zero variance**: Columns where >99% of values are identical (e.g. rare atom types), carrying zero predictive signal.\n",
            "3. **Collinearity / Multicollinearity**: Pairs of descriptors with $|r| > 0.95$ (e.g., number of valence electrons vs. molecular weight), which inflate standard errors in regression models.\n",
            "\n",
            "The `chemlib.descriptors.clean_descriptor_matrix` utility automates:\n",
            "- Dropping columns with >20% NaNs and median-imputing the rest.\n",
            "- Variance thresholding (default $\\sigma^2 > 0.01$).\n",
            "- Pairwise correlation pruning (dropping one descriptor from any pair with $|r| > 0.95$)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demonstrate clean_descriptor_matrix on a synthetic collinear/noisy dataset\n",
            "rng = np.random.RandomState(SEED)\n",
            "n_samples = 50\n",
            "\n",
            "f1 = rng.normal(10, 2, n_samples)                 # Informative\n",
            "f2 = f1 * 2.0 + rng.normal(0, 0.05, n_samples)    # Highly collinear with f1 (r > 0.99)\n",
            "f3 = np.ones(n_samples) * 5.0                     # Zero variance constant\n",
            "f4 = rng.normal(0, 1, n_samples)                  # Informative\n",
            "f4[0:3] = np.nan                                  # Has missing values\n",
            "\n",
            "df_raw_matrix = pd.DataFrame({\"feat_1\": f1, \"feat_collinear\": f2, \"feat_constant\": f3, \"feat_nan\": f4})\n",
            "print(\"Raw descriptor matrix shape:\", df_raw_matrix.shape)\n",
            "\n",
            "df_cleaned = descriptors.clean_descriptor_matrix(df_raw_matrix, variance_threshold=0.01, correlation_threshold=0.95)\n",
            "print(\"Cleaned descriptor matrix shape:\", df_cleaned.shape)\n",
            "print(\"Remaining columns:\", list(df_cleaned.columns))\n",
            "assert \"feat_constant\" not in df_cleaned.columns, \"Constant feature must be removed\"\n",
            "assert \"feat_collinear\" not in df_cleaned.columns, \"Collinear feature must be removed\""
        ]
    })

    # 13. Exercises
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Hands-On Exercises\n",
            "\n",
            "### Exercise 8.1: NumPy Implementation of First & Second Zagreb Indices\n",
            "Implement a function `calculate_zagreb_indices(mol: Chem.Mol) -> Tuple[int, int]` from first principles without using RDKit's built-in Zagreb functions.\n",
            "Use the adjacency matrix and vertex degrees $\\delta_i$:\n",
            "- $M_1 = \\sum_i \\delta_i^2$\n",
            "- $M_2 = \\sum_{(i,j) \\in E} \\delta_i \\delta_j = \\frac{1}{2} \\sum_{i} \\sum_{j} A_{ij} \\delta_i \\delta_j$"
        ]
    })

    if is_solution:
        code_8_1 = (
            "def calculate_zagreb_indices(mol: Chem.Mol) -> tuple:\n"
            "    # Complete working solution\n"
            "    A = Chem.GetAdjacencyMatrix(mol)\n"
            "    # Vertex degrees are the row sums of the heavy-atom adjacency matrix\n"
            "    degrees = np.sum(A, axis=1)\n"
            "    \n"
            "    # M1 = sum of degrees squared\n"
            "    m1 = int(np.sum(degrees ** 2))\n"
            "    \n"
            "    # M2 = sum over edges of d_i * d_j = 0.5 * sum_{i,j} A_{ij} * d_i * d_j\n"
            "    m2_matrix = A * np.outer(degrees, degrees)\n"
            "    m2 = int(np.sum(m2_matrix) // 2)\n"
            "    \n"
            "    return m1, m2\n"
            "\n"
            "m_test = Chem.MolFromSmiles('CCCCCC') # n-hexane\n"
            "print('Zagreb indices for n-hexane:', calculate_zagreb_indices(m_test))"
        )
    else:
        code_8_1 = (
            "def calculate_zagreb_indices(mol: Chem.Mol) -> tuple:\n"
            "    # YOUR CODE HERE: Compute and return (M1, M2) from the adjacency matrix and vertex degrees\n"
            "    raise NotImplementedError(\"Implement calculate_zagreb_indices\")\n"
            "\n"
            "# m_test = Chem.MolFromSmiles('CCCCCC')\n"
            "# print('Zagreb indices for n-hexane:', calculate_zagreb_indices(m_test))"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_8_1]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 8.1\n",
            "m_hexane = Chem.MolFromSmiles(\"CCCCCC\")\n",
            "m1_hex, m2_hex = calculate_zagreb_indices(m_hexane)\n",
            "# For linear n-hexane: degrees are 1, 2, 2, 2, 2, 1 -> sum(deg^2) = 1 + 4*4 + 1 = 18\n",
            "assert m1_hex == 18, f\"Expected M1=18 for n-hexane, got {m1_hex}\"\n",
            "# Edges: (1,2), (2,2), (2,2), (2,2), (2,1) -> 2 + 4 + 4 + 4 + 2 = 16\n",
            "assert m2_hex == 16, f\"Expected M2=16 for n-hexane, got {m2_hex}\"\n",
            "\n",
            "# For 2,2-dimethylbutane (CCC(C)(C)C)\n",
            "m_dmb = Chem.MolFromSmiles(\"CCC(C)(C)C\")\n",
            "m1_dmb, m2_dmb = calculate_zagreb_indices(m_dmb)\n",
            "assert m1_dmb == graphs.zagreb_m1_index(m_dmb)\n",
            "assert m2_dmb == graphs.zagreb_m2_index(m_dmb)\n",
            "print(\"✔ Exercise 8.1 passed!\")"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 8.2: Alkane Isomer Sub-Analysis ($C_5$ to $C_7$)\n",
            "Filter the alkane dataset for molecules having between 5 and 7 carbon atoms ($5 \\le n_C \\le 7$). Compute the Pearson correlation coefficient $r$ between the **Wiener Index** ($W$) and the experimental boiling point ($^\\circ\\text{C}$)."
        ]
    })

    if is_solution:
        code_8_2 = (
            "# Filter C5 to C7 alkanes from df_alkanes\n"
            "df_c5_c7 = df_alkanes[(df_alkanes['n_carbons'] >= 5) & (df_alkanes['n_carbons'] <= 7)].copy()\n"
            "r_wiener, _ = stats.pearsonr(df_c5_c7['wiener_index'], df_c5_c7['boiling_point_C'])\n"
            "print(f'Pearson r between Wiener Index and Boiling Point for C5-C7: {r_wiener:.4f}')"
        )
    else:
        code_8_2 = (
            "# Filter C5 to C7 alkanes from df_alkanes\n"
            "# YOUR CODE HERE: Compute r_wiener (Pearson r between wiener_index and boiling_point_C)\n"
            "df_c5_c7 = df_alkanes[(df_alkanes['n_carbons'] >= 5) & (df_alkanes['n_carbons'] <= 7)].copy()\n"
            "raise NotImplementedError(\"Compute r_wiener\")"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_8_2]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 8.2\n",
            "assert 'r_wiener' in locals(), \"r_wiener variable must be defined\"\n",
            "assert 0.80 < r_wiener < 1.0, f\"Expected strong positive correlation, got {r_wiener}\"\n",
            "print(\"✔ Exercise 8.2 passed!\")"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 8.3: Collinear Descriptor Filtering\n",
            "Given a descriptor matrix `df_sample` with 5 columns where column `col_A` and column `col_B` are collinear ($r = 0.99$), apply `clean_descriptor_matrix` with `correlation_threshold=0.90` and verify that exactly one of the collinear columns is removed."
        ]
    })

    if is_solution:
        code_8_3 = (
            "test_mat = pd.DataFrame({\n"
            "    'X1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],\n"
            "    'X2': [2.01, 3.99, 6.02, 7.98, 10.01, 11.99], # collinear with X1 (r > 0.99)\n"
            "    'X3': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],   # zero variance\n"
            "    'X4': [0.5, -0.2, 0.8, 1.2, -0.9, 0.3]\n"
            "})\n"
            "\n"
            "clean_result = descriptors.clean_descriptor_matrix(test_mat, variance_threshold=0.01, correlation_threshold=0.90)\n"
            "print('Remaining columns:', list(clean_result.columns))"
        )
    else:
        code_8_3 = (
            "test_mat = pd.DataFrame({\n"
            "    'X1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],\n"
            "    'X2': [2.01, 3.99, 6.02, 7.98, 10.01, 11.99],\n"
            "    'X3': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],\n"
            "    'X4': [0.5, -0.2, 0.8, 1.2, -0.9, 0.3]\n"
            "})\n"
            "\n"
            "# YOUR CODE HERE: Compute clean_result using descriptors.clean_descriptor_matrix\n"
            "raise NotImplementedError(\"Filter test_mat using clean_descriptor_matrix\")"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_8_3]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 8.3\n",
            "assert 'clean_result' in locals()\n",
            "assert 'X3' not in clean_result.columns, \"X3 has zero variance and must be dropped\"\n",
            "assert ('X1' in clean_result.columns) ^ ('X2' in clean_result.columns), \"Exactly one of X1 and X2 must be dropped\"\n",
            "assert 'X4' in clean_result.columns, \"X4 is informative and must be retained\"\n",
            "print(\"✔ Exercise 8.3 passed!\")"
        ]
    })

    # 14. Common Pitfalls
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Common Pitfalls\n",
            "\n",
            "1. **Calculating 3D Descriptors on 2D Structures**: Feeding a molecule generated from SMILES without generating a 3D conformer (`AllChem.EmbedMolecule`) to 3D descriptor routines (e.g. `CalcRadiusOfGyration`) either raises a runtime error or returns meaningless $0.0$ values.\n",
            "2. **Conformer Dependency**: 3D descriptors vary across different conformers of the same molecule. Never compute 3D descriptors on unoptimized conformers, and ensure a consistent minimization protocol (e.g. MMFF94) is documented.\n",
            "3. **Collinearity Inflation**: Using unpruned Mordred or PaDEL matrices in Ordinary Least Squares (OLS) regression causes severe variance inflation, erratic coefficients, and catastrophic failure on external test sets."
        ]
    })

    # 15. Summary
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Summary & Key Takeaways\n",
            "\n",
            "- **Taxonomy**: Descriptors scale from 0D (formula) to 4D (conformer ensembles). 2D descriptors are fast and conformer-independent, while 3D descriptors capture shape and surface area.\n",
            "- **Graph Invariants**: Topological indices (Zagreb, Randić, Wiener) condense molecular connectivity into single scalar invariants that correlate strongly with physical properties like boiling point ($R^2 > 0.95$).\n",
            "- **3D Shape**: NPR1 and NPR2 map molecules into rod, disc, and sphere triangular space.\n",
            "- **Rigorous Cleaning**: Curation via variance filtering, median imputation, and collinearity pruning ($|r| > 0.95$) is mandatory before machine learning modeling."
        ]
    })

    # 16. References
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 14. References & Further Reading\n",
            "\n",
            "- Todeschini, R.; Consonni, V. *Handbook of Molecular Descriptors*. Wiley-VCH, **2000**.\n",
            "- Randić, M. On characterization of molecular branching. *J. Am. Chem. Soc.* **1975**, 97, 6609–6615.\n",
            "- Wiener, H. Structural determination of paraffin boiling points. *J. Am. Chem. Soc.* **1947**, 69, 17–20.\n",
            "- Sauer, W. H. B.; Schwarz, M. K. Molecular shape diversity of combinatorial libraries: A graphical analysis. *J. Chem. Inf. Comput. Sci.* **2003**, 43, 987–1003.\n",
            "- Moriwaki, H. et al. Mordred: a molecular descriptor calculator. *J. Cheminform.* **2018**, 10, 4."
        ]
    })

    # 17. Final self-test cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Final self-test cell\n",
            "print(\"✔ NB08 checks passed\")"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

nb_student = create_nb08(is_solution=False)
nb_solution = create_nb08(is_solution=True)

out_student = Path("Cheminformatics/notebooks/nb08_molecular_descriptors.ipynb")
out_solution = Path("Cheminformatics/solutions/nb08_molecular_descriptors.ipynb")

out_student.parent.mkdir(parents=True, exist_ok=True)
out_solution.parent.mkdir(parents=True, exist_ok=True)

with open(out_student, "w") as f:
    json.dump(nb_student, f, indent=1)

with open(out_solution, "w") as f:
    json.dump(nb_solution, f, indent=1)

print("Generated nb08 student and solution notebooks!")
