import json, os, ast
from pathlib import Path

def create_nb09(is_solution=False):
    cells = []
    
    # 1. Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NB09 · Molecular Fingerprints: Keys, Paths, Circles, and Pharmacophore Pairs\n",
            "\n",
            "**Part C: Representations & Descriptors** · Estimated time: ~90 min · Difficulty: ●●○\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb09_fingerprints_keys_paths_circles.ipynb)\n",
            "\n",
            "---\n",
            "\n",
            "### One-Line Summary\n",
            "Master the modern RDKit `rdFingerprintGenerator` API, explore the four major fingerprint paradigms (substructure keys, path-based, circular ECFP/FCFP, and topological distance/pharmacophore pairs), investigate folding length and bit collisions, and decode bit environments using `bitInfo` to construct visual substructure galleries."
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
            "1. **Apply the Modern API**: Utilize `rdFingerprintGenerator` instead of deprecated legacy calls (addressing Course Errata E3).\n",
            "2. **Distinguish the 4 Fingerprint Families**: Contrast Substructure Keys (MACCS), Path-based (Daylight/RDKit/FP2), Circular (Morgan/ECFP4/ECFP6), and Distance-based (Atom Pairs & Topological Torsions).\n",
            "3. **Clarify ECFP Radius vs. Diameter**: Understand why radius 2 corresponds to diameter 4 (ECFP4) and radius 3 corresponds to diameter 6 (ECFP6).\n",
            "4. **Construct 2D Pharmacophore Signatures**: Explain the combinatorial distance-binning of `Pharm2D` Gobbi signatures and understand feature point geometry.\n",
            "5. **Analyze Hashing, Folding & Bit Collisions**: Quantify bit saturation and Tanimoto similarity distortion when folding fingerprints from 4096 down to 128 bits.\n",
            "6. **Map Bits to Substructures**: Extract `bitInfo` from Morgan fingerprints and render a visual bit-to-substructure gallery explaining which chemical fragments activate individual bits."
        ]
    })

    # 3. Prerequisites
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Prerequisites & Lecture Links\n",
            "\n",
            "- **Prerequisites**: SMILES strings (`nb01`, `nb03`), molecular graphs (`nb04`), and functional groups (`nb07`).\n",
            "- **Lecture Aligned**: Lecture Deck 06 (*Similarity Search & Fingerprints*), Lecture Deck 04 (*Organic Structures*), and Lecture Deck 03 (*Cheminformatics Tools*).\n",
            "- **Errata Covered**:\n",
            "  - **E3**: Deprecation of `GetMorganFingerprintAsBitVect`; corrected radius terminology (radius 2 = ECFP4, radius 3 = ECFP6).\n",
            "  - **E4**: Open Babel FP2 bit length clarification (1,024 bits vs 256)."
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
            "import os, sys, random, warnings\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from rdkit import Chem, RDLogger, DataStructs\n",
            "from rdkit.Chem import AllChem, Draw, MACCSkeys, rdFingerprintGenerator\n",
            "from rdkit.Chem.Draw import rdMolDraw2D\n",
            "\n",
            "# Ensure src directory is in sys.path\n",
            "repo_root = Path.cwd().parent if Path.cwd().name in [\"notebooks\", \"solutions\"] else Path.cwd()\n",
            "if str(repo_root / \"src\") not in sys.path:\n",
            "    sys.path.insert(0, str(repo_root / \"src\"))\n",
            "\n",
            "import chemlib\n",
            "from chemlib import fingerprints\n",
            "\n",
            "SEED = 42\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
            "RDLogger.DisableLog(\"rdApp.*\")\n",
            "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "print(\"Environment initialized successfully. chemlib version:\", chemlib.__version__)"
        ]
    })

    # 5. Taxonomy of Fingerprints
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. The Molecular Fingerprint Paradigm\n",
            "\n",
            "A **molecular fingerprint** is a fixed-length or hashed bit vector (or count vector) representing the chemical features of a molecule. Fingerprints enable ultra-fast boolean operations (bitwise `AND`, `OR`, `POPCNT`), allowing millions of pairwise similarity comparisons per second.\n",
            "\n",
            "### Fingerprint Taxonomy Table\n",
            "\n",
            "| Family | Algorithm / Format | Typical Bit Length | Hashed? | Reversible to Pattern? | Primary Application |\n",
            "|:---|:---|:---|:---|:---|:---|\n",
            "| **Substructure Keys** | **MACCS Keys** | 166 bits | No | Yes (predefined SMARTS dictionary) | Coarse screening, chemical diversity profiling |\n",
            "| **Substructure Keys** | **PubChem Keys** | 881 bits | No | Yes (explicit dictionary) | Broad public screening, biological activity clustering |\n",
            "| **Path-Based** | **RDKit Topological** | 2048 bits | Yes | No (collisions occur) | Linear and branched substructure matching |\n",
            "| **Path-Based** | **Open Babel FP2** | 1024 bits | Yes | No | Fast pre-filtering (linear paths 1–7 atoms) |\n",
            "| **Circular (ECFP4)** | **Morgan (radius 2)** | 2048 bits | Yes | No (can map to atom env via `bitInfo`) | Gold standard for ligand-based virtual screening & QSAR |\n",
            "| **Circular (ECFP6)** | **Morgan (radius 3)** | 2048 bits | Yes | No | Larger macrocyclic / natural product environments |\n",
            "| **Distance-Based** | **Atom Pairs** | 2048 bits | Yes | Partial (atom types + path distance) | Scaffold hopping, distant analog retrieval |\n",
            "| **Distance-Based** | **Topological Torsions** | 2048 bits | Yes | Partial (4-atom bonded sequences) | Conformationally relevant local geometry |\n",
            "| **Pharmacophore** | **2D Pharm2D (Gobbi)** | 39,973 bits | Combinatorial | Yes (distance bins between pharmacophoric points) | Receptor-independent pharmacophore matching |"
        ]
    })

    # 6. Modern rdFingerprintGenerator API & Errata E3
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. The Modern `rdFingerprintGenerator` API & Errata E3\n",
            "\n",
            "> [!IMPORTANT]\n",
            "> **Lecture Errata (E3) & Modern Best Practice**:\n",
            "> In historical slides and older RDKit tutorials, circular fingerprints were generated via:\n",
            "> `AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)`\n",
            "> **This API is deprecated.** Modern RDKit code uses the unified `rdFingerprintGenerator` framework, which provides reusable, thread-safe generator objects with consistent configuration.\n",
            ">\n",
            "> **Radius vs. Diameter Terminology**:\n",
            "> - **Radius 1** = Diameter 2 = **ECFP2**\n",
            "> - **Radius 2** = Diameter 4 = **ECFP4** (the standard)\n",
            "> - **Radius 3** = Diameter 6 = **ECFP6**\n",
            "> *Never set radius=3 when ECFP4 is intended!*"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instantiate modern fingerprint generators\n",
            "fpgen_maccs = MACCSkeys.GenMACCSKeys                                                    # Substructure keys (166)\n",
            "fpgen_rdkit = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=2048) # Path-based\n",
            "fpgen_ecfp4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)          # Circular ECFP4\n",
            "fpgen_ecfp6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)          # Circular ECFP6\n",
            "fpgen_atompair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)              # Atom Pairs\n",
            "fpgen_torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)    # Topological Torsions\n",
            "\n",
            "print(\"Fingerprint generators constructed successfully.\")"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare fingerprint bit density (number of ON bits) on Aspirin\n",
            "aspirin = Chem.MolFromSmiles(\"CC(=O)Oc1ccccc1C(=O)O\")\n",
            "\n",
            "fp_maccs = MACCSkeys.GenMACCSKeys(aspirin)\n",
            "fp_rdkit = fpgen_rdkit.GetFingerprint(aspirin)\n",
            "fp_ecfp4 = fpgen_ecfp4.GetFingerprint(aspirin)\n",
            "fp_ecfp6 = fpgen_ecfp6.GetFingerprint(aspirin)\n",
            "fp_ap = fpgen_atompair.GetFingerprint(aspirin)\n",
            "fp_tt = fpgen_torsion.GetFingerprint(aspirin)\n",
            "\n",
            "fp_stats = [\n",
            "    (\"MACCS Keys\", fp_maccs.GetNumBits(), fp_maccs.GetNumOnBits()),\n",
            "    (\"RDKit Topological\", fp_rdkit.GetNumBits(), fp_rdkit.GetNumOnBits()),\n",
            "    (\"Morgan ECFP4 (radius=2)\", fp_ecfp4.GetNumBits(), fp_ecfp4.GetNumOnBits()),\n",
            "    (\"Morgan ECFP6 (radius=3)\", fp_ecfp6.GetNumBits(), fp_ecfp6.GetNumOnBits()),\n",
            "    (\"Atom Pairs\", fp_ap.GetNumBits(), fp_ap.GetNumOnBits()),\n",
            "    (\"Topological Torsions\", fp_tt.GetNumBits(), fp_tt.GetNumOnBits()),\n",
            "]\n",
            "\n",
            "df_stats = pd.DataFrame(fp_stats, columns=[\"Fingerprint_Type\", \"Total_Bits\", \"ON_Bits\"])\n",
            "df_stats[\"Bit_Density_%\"] = (df_stats[\"ON_Bits\"] / df_stats[\"Total_Bits\"]) * 100.0\n",
            "print(\"Aspirin Fingerprint Profile:\")\n",
            "df_stats"
        ]
    })

    # 7. 2D Pharmacophore Fingerprints
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. 2D Pharmacophore Fingerprints: Gobbi Signatures\n",
            "\n",
            "Pharmacophore fingerprints represent molecules not by atomic connectivity, but by spatial or topological relationships between **pharmacophoric feature points**:\n",
            "- **H-Bond Acceptor (HBA)**\n",
            "- **H-Bond Donor (HBD)**\n",
            "- **Positive Ionizable / Basic (PI)**\n",
            "- **Negative Ionizable / Acidic (NI)**\n",
            "- **Aromatic / Hydrophobic (AR)**\n",
            "\n",
            "### Combinatorial Arithmetic\n",
            "In a **3-point pharmacophore** (triplet of features), each feature pair is separated by a topological distance binned into discrete intervals (e.g., 2–3 bonds, 4–5 bonds, 6–7 bonds, etc.).\n",
            "- A small toy system with 2 feature patterns, 2–3 point triangles, and 2 distance bins produces $6 + 32 = 38$ bits.\n",
            "- The full **Gobbi `Pharm2D` signature** implemented in RDKit contains **39,973 bits**, capturing all 2-point and 3-point pharmacophoric triangles across 8 feature classes and 7 distance bins!"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute Gobbi 2D Pharmacophore fingerprint for Aspirin and Caffeine\n",
            "from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D\n",
            "\n",
            "caffeine = Chem.MolFromSmiles(\"Cn1cnc2c1c(=O)n(C)c(=O)n2C\")\n",
            "\n",
            "pharm_asp = Generate.Gen2DFingerprint(aspirin, Gobbi_Pharm2D.factory)\n",
            "pharm_caff = Generate.Gen2DFingerprint(caffeine, Gobbi_Pharm2D.factory)\n",
            "\n",
            "print(f\"Gobbi Pharm2D Signature total bit size: {pharm_asp.GetNumBits()}\")\n",
            "print(f\"Aspirin pharmacophore ON bits: {pharm_asp.GetNumOnBits()}\")\n",
            "print(f\"Caffeine pharmacophore ON bits: {pharm_caff.GetNumOnBits()}\")\n",
            "\n",
            "sim_pharm = DataStructs.TanimotoSimilarity(pharm_asp, pharm_caff)\n",
            "print(f\"Pharm2D Tanimoto Similarity between Aspirin and Caffeine: {sim_pharm:.4f}\")"
        ]
    })

    # 8. Hashing, Folding, and Bit Collision Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Hashing, Folding, and Bit Collision Analysis\n",
            "\n",
            "When circular substructure environments are generated, they are assigned a 32-bit integer hash code ($0$ to $2^{32}-1$). Because storing $4.3 \\times 10^9$ bits per molecule is memory-prohibitive, the integer is **folded** into a fixed-length bit vector of length $L$ via modulo division:\n",
            "$$\\text{bit\\_index} = \\text{hash\\_code} \\pmod L$$\n",
            "\n",
            "### The Bit Collision Phenomenon\n",
            "If two chemically distinct fragments happen to satisfy $\\text{hash}_1 \\equiv \\text{hash}_2 \\pmod L$, they set the **exact same bit position**.\n",
            "- At large lengths ($L = 4096$ or $2048$), collisions are rare.\n",
            "- At small lengths ($L \\le 256$), bit collisions multiply, causing **saturation** (too many bits ON) and artificially **inflating Tanimoto similarity** between unrelated molecules!"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Folding experiment: evaluate bit density and similarity distortion from 4096 down to 128 bits\n",
            "salicylic_acid = Chem.MolFromSmiles(\"Oc1ccccc1C(=O)O\") # structurally related analog\n",
            "\n",
            "bit_sizes = [4096, 2048, 1024, 512, 256, 128]\n",
            "folding_results = []\n",
            "\n",
            "for n_bits in bit_sizes:\n",
            "    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)\n",
            "    fp_a = gen.GetFingerprint(aspirin)\n",
            "    fp_s = gen.GetFingerprint(salicylic_acid)\n",
            "    \n",
            "    tan = DataStructs.TanimotoSimilarity(fp_a, fp_s)\n",
            "    density = (fp_a.GetNumOnBits() / n_bits) * 100.0\n",
            "    folding_results.append({\n",
            "        \"Bit_Length\": n_bits,\n",
            "        \"Aspirin_ON_Bits\": fp_a.GetNumOnBits(),\n",
            "        \"Bit_Density_%\": round(density, 2),\n",
            "        \"Tanimoto_Similarity\": round(tan, 4)\n",
            "    })\n",
            "\n",
            "df_folding = pd.DataFrame(folding_results)\n",
            "print(\"Impact of Folding Length on Bit Density & Tanimoto Similarity:\")\n",
            "df_folding"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot bit density and similarity as a function of bit size\n",
            "fig, ax1 = plt.subplots(figsize=(7.5, 4.5))\n",
            "\n",
            "color = '#1f77b4'\n",
            "ax1.set_xlabel('Fingerprint Length (Bits)', fontweight='bold')\n",
            "ax1.set_ylabel('Bit Density (%)', color=color, fontweight='bold')\n",
            "ax1.plot(df_folding['Bit_Length'], df_folding['Bit_Density_%'], color=color, marker='o', lw=2)\n",
            "ax1.tick_params(axis='y', labelcolor=color)\n",
            "ax1.set_xscale('log', base=2)\n",
            "\n",
            "ax2 = ax1.twinx()\n",
            "color = '#d62728'\n",
            "ax2.set_ylabel('Tanimoto Similarity (Aspirin vs Salicylic Acid)', color=color, fontweight='bold')\n",
            "ax2.plot(df_folding['Bit_Length'], df_folding['Tanimoto_Similarity'], color=color, marker='s', lw=2, linestyle='--')\n",
            "ax2.tick_params(axis='y', labelcolor=color)\n",
            "\n",
            "plt.title('Effect of Fingerprint Folding: Saturation & Similarity Inflation', fontsize=12, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 9. Explainable AI: Bit Info & Substructure Galleries
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Explainable AI: Mapping Bits to Atomic Environments (`bitInfo`)\n",
            "\n",
            "A frequent criticism of circular fingerprints in medicinal chemistry is that they are \"black boxes\". However, RDKit allows full inspection of the exact atomic substructure responsible for setting any bit via the **`bitInfo` map**:\n",
            "\n",
            "$$\\text{bit\\_id} \\longrightarrow [(\\text{atom\\_idx}_1, \\text{radius}_1), (\\text{atom\\_idx}_2, \\text{radius}_2), \\dots]$$\n",
            "\n",
            "- **Radius 0**: Single atom with its atomic number, degree, and charge (e.g. carbonyl oxygen).\n",
            "- **Radius 1**: Central atom plus its immediate bonded neighbors (e.g. ester carbonyl $O=C-O$).\n",
            "- **Radius 2**: Central atom, immediate neighbors, and next-nearest neighbors (diameter = 4 bonds)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Extract bitInfo using AdditionalOutput in modern RDKit\n",
            "ao = rdFingerprintGenerator.AdditionalOutput()\n",
            "ao.AllocateBitInfoMap()\n",
            "\n",
            "fp_aspirin_info = fpgen_ecfp4.GetFingerprint(aspirin, additionalOutput=ao)\n",
            "bit_info = ao.GetBitInfoMap()\n",
            "\n",
            "print(f\"Total unique bits set in Aspirin: {len(bit_info)}\")\n",
            "# Display the first 5 bits and their atom environments\n",
            "for bit_id in list(bit_info.keys())[:5]:\n",
            "    atom_radius_pairs = bit_info[bit_id]\n",
            "    print(f\"  Bit {bit_id:4d} -> Set by {len(atom_radius_pairs)} center(s): {atom_radius_pairs}\")"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Render the substructure environment for the first 3 bits in Aspirin\n",
            "sample_bits = list(bit_info.keys())[:3]\n",
            "print(\"Visualizing 3 sample Morgan bit environments in Aspirin:\")\n",
            "for b in sample_bits:\n",
            "    center_atom, radius = bit_info[b][0]\n",
            "    print(f\"Bit {b}: Central atom = {aspirin.GetAtomWithIdx(center_atom).GetSymbol()} (idx {center_atom}), Radius = {radius}\")\n",
            "\n",
            "# Draw the bits\n",
            "img_bits = Draw.DrawMorganBits(\n",
            "    [(aspirin, b, bit_info) for b in sample_bits],\n",
            "    molsPerRow=3,\n",
            "    subImgSize=(220, 180),\n",
            "    legends=[f\"Bit {b} (rad {bit_info[b][0][1]})\" for b in sample_bits]\n",
            ")\n",
            "img_bits"
        ]
    })

    # 10. Exercises
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Hands-On Exercises\n",
            "\n",
            "### Exercise 9.1: Multi-Fingerprint Similarity Benchmark\n",
            "Write a function `compute_cross_similarities(mol1: Chem.Mol, mol2: Chem.Mol) -> dict` that computes the Tanimoto similarity between `mol1` and `mol2` across 4 different fingerprint types:\n",
            "1. MACCS Keys\n",
            "2. RDKit Topological (2048 bits)\n",
            "3. Morgan ECFP4 (radius=2, 2048 bits)\n",
            "4. Atom Pairs (2048 bits)"
        ]
    })

    if is_solution:
        code_9_1 = (
            "def compute_cross_similarities(mol1: Chem.Mol, mol2: Chem.Mol) -> dict:\n"
            "    # Complete working solution\n"
            "    # 1. MACCS\n"
            "    fp1_maccs = MACCSkeys.GenMACCSKeys(mol1)\n"
            "    fp2_maccs = MACCSkeys.GenMACCSKeys(mol2)\n"
            "    tan_maccs = DataStructs.TanimotoSimilarity(fp1_maccs, fp2_maccs)\n"
            "    \n"
            "    # 2. RDKit topological\n"
            "    fp1_rdkit = fpgen_rdkit.GetFingerprint(mol1)\n"
            "    fp2_rdkit = fpgen_rdkit.GetFingerprint(mol2)\n"
            "    tan_rdkit = DataStructs.TanimotoSimilarity(fp1_rdkit, fp2_rdkit)\n"
            "    \n"
            "    # 3. ECFP4\n"
            "    fp1_ecfp4 = fpgen_ecfp4.GetFingerprint(mol1)\n"
            "    fp2_ecfp4 = fpgen_ecfp4.GetFingerprint(mol2)\n"
            "    tan_ecfp4 = DataStructs.TanimotoSimilarity(fp1_ecfp4, fp2_ecfp4)\n"
            "    \n"
            "    # 4. Atom pairs\n"
            "    fp1_ap = fpgen_atompair.GetFingerprint(mol1)\n"
            "    fp2_ap = fpgen_atompair.GetFingerprint(mol2)\n"
            "    tan_ap = DataStructs.TanimotoSimilarity(fp1_ap, fp2_ap)\n"
            "    \n"
            "    return {\n"
            "        'MACCS': float(tan_maccs),\n"
            "        'RDKit': float(tan_rdkit),\n"
            "        'ECFP4': float(tan_ecfp4),\n"
            "        'AtomPair': float(tan_ap)\n"
            "    }\n"
            "\n"
            "sims_demo = compute_cross_similarities(aspirin, caffeine)\n"
            "print('Cross-fingerprint similarities between Aspirin and Caffeine:', sims_demo)"
        )
    else:
        code_9_1 = (
            "def compute_cross_similarities(mol1: Chem.Mol, mol2: Chem.Mol) -> dict:\n"
            "    # YOUR CODE HERE: Compute Tanimoto similarity for MACCS, RDKit, ECFP4, and AtomPair\n"
            "    raise NotImplementedError(\"Implement compute_cross_similarities\")\n"
            "\n"
            "# sims_demo = compute_cross_similarities(aspirin, caffeine)\n"
            "# print('Cross-fingerprint similarities:', sims_demo)"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_9_1]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 9.1\n",
            "sims_res = compute_cross_similarities(aspirin, caffeine)\n",
            "assert \"MACCS\" in sims_res and \"RDKit\" in sims_res\n",
            "assert \"ECFP4\" in sims_res and \"AtomPair\" in sims_res\n",
            "for k, v in sims_res.items():\n",
            "    assert 0.0 <= v <= 1.0, f\"Similarity {k} must be in [0, 1], got {v}\"\n",
            "# Aspirin and caffeine are chemically distinct; ECFP4 should be low\n",
            "assert sims_res[\"ECFP4\"] < 0.25, f\"Expected low ECFP4 similarity, got {sims_res['ECFP4']}\"\n",
            "print(\"✔ Exercise 9.1 passed!\")"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 9.2: Substructure Bit Extraction\n",
            "Using `bit_info` for Aspirin, write code to find all bits that have a circular radius of exactly $0$ (individual heavy atoms) and count how many unique radius-0 bits exist."
        ]
    })

    if is_solution:
        code_9_2 = (
            "# Complete working solution\n"
            "radius_0_bits = [bit_id for bit_id, envs in bit_info.items() if any(r == 0 for atom_idx, r in envs)]\n"
            "n_radius_0 = len(radius_0_bits)\n"
            "print(f'Found {n_radius_0} unique radius-0 bits in Aspirin: {radius_0_bits}')"
        )
    else:
        code_9_2 = (
            "# YOUR CODE HERE: Find all bit IDs in bit_info having radius == 0 and store count in n_radius_0\n"
            "radius_0_bits = []\n"
            "n_radius_0 = 0\n"
            "raise NotImplementedError(\"Extract radius-0 bits\")"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_9_2]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 9.2\n",
            "assert n_radius_0 > 0, \"Must find at least one radius-0 bit\"\n",
            "# Aspirin heavy atoms: 9 carbons (aliphatic, aromatic, carbonyl) + 4 oxygens -> typically 5-7 distinct atom types\n",
            "assert 4 <= n_radius_0 <= 10, f\"Expected 4 to 10 distinct radius-0 bits, got {n_radius_0}\"\n",
            "print(\"✔ Exercise 9.2 passed!\")"
        ]
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 9.3: Folding & Collision Drift\n",
            "Compute the Tanimoto similarity between Aspirin and Salicylic acid using an ECFP4 generator folded to **64 bits**. Verify that due to severe bit collisions, the 64-bit Tanimoto similarity is significantly greater than the 2048-bit Tanimoto similarity."
        ]
    })

    if is_solution:
        code_9_3 = (
            "# Complete working solution\n"
            "gen_64 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=64)\n"
            "fp_asp_64 = gen_64.GetFingerprint(aspirin)\n"
            "fp_sal_64 = gen_64.GetFingerprint(salicylic_acid)\n"
            "tan_64 = float(DataStructs.TanimotoSimilarity(fp_asp_64, fp_sal_64))\n\n"
            "gen_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)\n"
            "fp_asp_2048 = gen_2048.GetFingerprint(aspirin)\n"
            "fp_sal_2048 = gen_2048.GetFingerprint(salicylic_acid)\n"
            "tan_2048 = float(DataStructs.TanimotoSimilarity(fp_asp_2048, fp_sal_2048))\n\n"
            "print(f'Tanimoto at 2048 bits: {tan_2048:.4f}')\n"
            "print(f'Tanimoto at 64 bits:   {tan_64:.4f}')"
        )
    else:
        code_9_3 = (
            "# YOUR CODE HERE: Compute tan_64 and tan_2048 between aspirin and salicylic_acid\n"
            "tan_64 = 0.0\n"
            "tan_2048 = 0.0\n"
            "raise NotImplementedError(\"Compute tan_64 and tan_2048\")"
        )

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_9_3]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Self-check test for Exercise 9.3\n",
            "assert tan_64 > tan_2048, f\"Expected collision-induced inflation (tan_64 > tan_2048), got {tan_64} vs {tan_2048}\"\n",
            "print(\"✔ Exercise 9.3 passed!\")"
        ]
    })

    # 11. Common Pitfalls
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Common Pitfalls\n",
            "\n",
            "1. **ECFP Radius / Diameter Confusion**: Specifying `radius=4` in `GetMorganGenerator` generates **ECFP8** (an 8-bond diameter neighborhood), which is almost universally overfitted and sparse. Standard ECFP4 requires `radius=2`.\n",
            "2. **Comparing Dissimilar Fingerprint Lengths**: Computing Tanimoto similarity between a 1024-bit fingerprint and a 2048-bit fingerprint will throw a dimension mismatch error in C++.\n",
            "3. **Using Bit Vectors for Repeated Features**: In bit vectors, having 1 hydroxyl group vs 5 hydroxyl groups sets the exact same bit (bit is ON). When stoichiometry matters, use count fingerprints (`GetCountFingerprint`)."
        ]
    })

    # 12. Summary
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Summary & Key Takeaways\n",
            "\n",
            "- **Modern RDKit**: Always use `rdFingerprintGenerator` for circular, path-based, and atom-pair fingerprints.\n",
            "- **Four Paradigms**: Substructure keys (fixed keys), Path-based (linear subgraphs), Circular (radial Morgan environments), and Distance-based (atom pairs/pharmacophores).\n",
            "- **Bit Collisions & Folding**: Folding compresses memory at the cost of bit collisions; lengths below 512 bits suffer from severe Tanimoto inflation.\n",
            "- **Explainability**: `bitInfo` maps Morgan bits directly back to atom indices and radii, converting abstract bit vectors into interpretable visual substructures."
        ]
    })

    # 13. References
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. References & Further Reading\n",
            "\n",
            "- Rogers, D.; Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754.\n",
            "- Durant, J. L. et al. Reoptimization of MDL keys for use in drug discovery. *J. Chem. Inf. Comput. Sci.* **2002**, 42, 1273–1280.\n",
            "- Gobbi, A.; Lee, M. L. PHRP: Pharmacophore fingerprinting of small molecules. *J. Chem. Inf. Comput. Sci.* **2003**, 43, 317–323.\n",
            "- Carhart, R. E. et al. Atom pairs as molecular features in structure-activity studies. *J. Chem. Inf. Comput. Sci.* **1985**, 25, 64–73."
        ]
    })

    # 14. Final self-test cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Final self-test cell\n",
            "print(\"✔ NB09 checks passed\")"
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

nb_student = create_nb09(is_solution=False)
nb_solution = create_nb09(is_solution=True)

out_student = Path("Cheminformatics/notebooks/nb09_fingerprints_keys_paths_circles.ipynb")
out_solution = Path("Cheminformatics/solutions/nb09_fingerprints_keys_paths_circles.ipynb")

out_student.parent.mkdir(parents=True, exist_ok=True)
out_solution.parent.mkdir(parents=True, exist_ok=True)

with open(out_student, "w") as f:
    json.dump(nb_student, f, indent=1)

with open(out_solution, "w") as f:
    json.dump(nb_solution, f, indent=1)

print("Generated nb09 student and solution notebooks!")
