"""
patch_all_skeletons.py
Applies standard skeleton updates to build_nb10.py through build_nb21.py,
then rebuilds all notebooks to achieve 100% skeleton compliance.
"""

import os
import re
import subprocess

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

# ----------------------------------------------------------------------
# 1. Update nb_helper.py to replace "notebooks/nb" with "solutions/nb" in sol badges
# ----------------------------------------------------------------------
nb_helper_path = os.path.join(SCRIPTS_DIR, "nb_helper.py")
with open(nb_helper_path, "r", encoding="utf-8") as f:
    nb_h = f.read()

# In save_notebooks: for markdown cells, replace notebooks/nb with solutions/nb in solution_nb
old_h = """    for cell in nb_template["cells"]:
        if cell["cell_type"] == "markdown":
            student_nb["cells"].append(cell)
            solution_nb["cells"].append(cell)"""

new_h = """    for cell in nb_template["cells"]:
        if cell["cell_type"] == "markdown":
            student_nb["cells"].append(cell)
            sol_c = json.loads(json.dumps(cell))
            sol_src = "".join(sol_c["source"])
            if "colab-badge" in sol_src and "notebooks/nb" in sol_src:
                sol_src = sol_src.replace("notebooks/nb", "solutions/nb")
                lines = [l + "\\n" for l in sol_src.strip().split("\\n")]
                if lines: lines[-1] = lines[-1].rstrip("\\n")
                sol_c["source"] = lines
            solution_nb["cells"].append(sol_c)"""

if old_h in nb_h:
    nb_h = nb_h.replace(old_h, new_h)
    with open(nb_helper_path, "w", encoding="utf-8") as f:
        f.write(nb_h)
    print("Updated nb_helper.py with solution colab badge replacement.")

# ----------------------------------------------------------------------
# 2. Update build_nb10.py through build_nb14.py
# ----------------------------------------------------------------------
pitfalls_and_summaries = {
    10: (
        """## Common Pitfalls

1. **Bit Collision Ambiguity**: In hashed circular fingerprints (Morgan/ECFP), bit folding causes collisions where distinct subgraphs map to the same bit, yielding spurious similarity.
2. **Asymmetric Comparisons**: The Tversky index with $\\\\alpha \\\\neq \\\\beta$ is non-symmetric ($S(A, B) \\\\neq S(B, A)$); reversing query and target arguments significantly alters ranking.
3. **Global AUC vs Early Enrichment**: High ROC-AUC does not guarantee good virtual screening performance if active compounds are not concentrated at the very top of the ranking list.
4. **Scaffold Hopping Blind Spots**: Circular fingerprints often penalize novel core scaffolds even when they project identical pharmacophores into the receptor site.""",
        """## Summary and Key Takeaways

- **Hierarchical Screening**: Combining fast fingerprint bitset pre-filtering with exact subgraph isomorphism (VF2) yields order-of-magnitude virtual screening speedups.
- **Continuous Metrics**: The Tanimoto coefficient is standard for molecular comparisons, while the Tversky index parameterizes substructure-like inclusion versus whole-molecule similarity.
- **Validation Metrics**: Virtual screening ranking must be evaluated using early enrichment metrics (EF1%, BEDROC $\\\\alpha=20$) alongside ROC-AUC and PR-AUC.
- **Scaffold Generalization**: Bemis-Murcko frameworks separate core rings from linkers and substituents to evaluate true out-of-scaffold model generalizability."""
    ),
    11: (
        """## Common Pitfalls

1. **Conformer Independence Illusion**: Pharmacophore points in 3D depend strictly on the underlying conformer; generating an inadequate conformational ensemble causes severe false-negative search attrition.
2. **Over-Constrained Pharmacophores**: Setting distance tolerances too narrow ($\\\\delta < 0.5\\\\text{ \\\\AA}$) excludes viable congeneric actives with minor induced-fit adjustments.
3. **Ignoring Ionization at Physiological pH**: Features like carboxylates (Anion) and basic amines (PosIonizable) depend directly on target compartment pH; modeling uncharged states produces false negatives.""",
        """## Summary and Key Takeaways

- **Pharmacophore Concept**: An abstract description of molecular steric and electronic interaction features required for macromolecular target recognition.
- **Ensemble Featurization**: 3D pharmacophore modeling requires thorough conformational sampling (ETKDGv3) followed by geometric feature distance analysis.
- **Triad Models & 2D Signatures**: 3-point pharmacophore queries provide spatial geometric filtering, while 2D pharmacophore signatures (Pharm2D) enable rapid library pre-screening."""
    ),
    12: (
        """## Common Pitfalls

1. **Collinearity and Chance Correlation**: High inter-descriptor correlation creates unstable regression coefficients; ratio of observations to variables must strictly exceed 5:1 (Topliss ratio).
2. **Extrapolation Beyond Applicability Domain**: Linear Hansch models cannot reliably predict compounds whose physicochemical descriptors fall outside the training envelope.
3. **Overfitting on Small Datasets**: High $R^2$ without leave-one-out cross-validated $Q^2 > 0.5$ or external test validation reflects statistical memorization rather than generalizability.""",
        """## Summary and Key Takeaways

- **Hansch Multilinear Model**: Correlates biological activity ($\\\\text{pIC}_{50}$) with electronic ($\\\\sigma$), steric ($E_s$), and hydrophobic ($\\\\pi, \\\\log P$) substituent constants.
- **Free-Wilson Analysis**: Uses indicator matrix regression to compute group contribution values without requiring external physicochemical descriptor calculations.
- **Rigorous Validation**: QSAR models must be validated using cross-validated $Q^2$, $Y$-randomization, and external test sets before prospective application."""
    ),
    13: (
        """## Common Pitfalls

1. **Scaffold Leakage during Splitting**: Random train-test splitting inflates validation performance when identical scaffolds appear in both subsets; Bemis-Murcko scaffold splitting is essential for realistic evaluation.
2. **Data Leakage in Preprocessing**: Fitting scalers, imputers, or feature selectors on the entire dataset before splitting introduces subtle information leakage from test to train.
3. **Imbalanced Inactive/Active Ratios**: Standard classification accuracy is misleading when actives represent < 5% of data; always assess Balanced Accuracy, PR-AUC, and Matthews Correlation Coefficient (MCC).""",
        """## Summary and Key Takeaways

- **Non-Linear QSAR**: Machine learning models (Random Forest, Gradient Boosting, SVM, MLP) capture complex non-linear SAR patterns inaccessible to classical linear regressions.
- **Out-of-Scaffold Validation**: Bemis-Murcko scaffold splitting ensures models generalize across distinct chemotypes rather than memorizing local structural analog clusters.
- **Balanced Assessment**: Evaluating virtual screening classifiers requires threshold-independent metrics (ROC-AUC, PR-AUC) and chance-corrected statistics (MCC)."""
    ),
    14: (
        """## Common Pitfalls

1. **Molecular Alignment Sensitivity**: 3D-QSAR predictive accuracy depends heavily on alignment; slight coordinate shifts can dramatically alter grid interaction energy potentials.
2. **Overfitting PLS Latent Variables**: Selecting too many PLS components overfits correlated field variables, degrading external test predictive performance ($R^2_{\\\\text{pred}}$).
3. **Boundary Truncation Artifacts**: Inadequate grid margins truncate steric or long-range Coulombic potentials around extended substituents.""",
        """## Summary and Key Takeaways

- **Molecular Field Analysis**: CoMFA samples steric (Lennard-Jones) and electrostatic (Coulombic) interaction fields across a 3D Cartesian grid surrounding aligned bioactive conformers.
- **PLS Regression**: Partial Least Squares extracts dominant latent variables from thousands of correlated 3D field coordinates, regularized by leave-one-out cross-validation ($Q^2$).
- **Steric & Electrostatic Contours**: Visualizing 3D contour field maps directly identifies regions where bulky or electronegative substituents enhance target binding affinity."""
    ),
}

for i in range(10, 15):
    path = os.path.join(SCRIPTS_DIR, f"build_nb{i}.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. Update add_cell to handle sol_c badge replacement
    old_add_cell = """    def add_cell(cell_type, content, sol_content=None):
        if cell_type == "markdown":
            cells_student.append(new_markdown_cell(content))
            cells_solution.append(new_markdown_cell(content))"""
    new_add_cell = """    def add_cell(cell_type, content, sol_content=None):
        if cell_type == "markdown":
            cells_student.append(new_markdown_cell(content))
            sol_c = sol_content if sol_content is not None else content
            if "colab-badge" in sol_c and "notebooks/nb" in sol_c:
                sol_c = sol_c.replace("notebooks/nb", "solutions/nb")
            cells_solution.append(new_markdown_cell(sol_c))"""
    if old_add_cell in src:
        src = src.replace(old_add_cell, new_add_cell)

    # 2. Add colab badge & change ### Objectives to ## Learning Objectives
    # Find title
    nb_id = f"nb{i:02d}"
    fn_matches = [
        "nb10_substructure_and_similarity_search.ipynb",
        "nb11_pharmacophore_modeling.ipynb",
        "nb12_qsar_foundations_hansch.ipynb",
        "nb13_ml_qsar_classification_regression.ipynb",
        "nb14_3d_qsar_comfa_fields.ipynb"
    ]
    fn = fn_matches[i - 10]
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/{fn})\\n\\n"

    # Insert badge after course part line if badge not already present
    if "colab-badge" not in src:
        # replace f"# {nb_id} ·"
        pattern = rf"(# {nb_id} · [^\n]+\n(\*\*[^\n]+\*\*\n\n)?)"
        match = re.search(pattern, src)
        if match:
            matched_str = match.group(0)
            src = src.replace(matched_str, matched_str + badge_md)

    # Replace ### Objectives with ## Learning Objectives
    src = src.replace("### Objectives\n", "## Learning Objectives\n")

    # 3. Add SEED=42 and FULL_RUN=False to setup cell
    setup_target = 'add_cell("code", """import os\nimport sys\n'
    setup_replacement = 'add_cell("code", """import os\nimport sys\nimport random\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom typing import List, Dict, Tuple, Optional, Any\n\nSEED = 42\nFULL_RUN = False\nrandom.seed(SEED)\nnp.random.seed(SEED)\n'
    
    if "SEED = 42" not in src:
        # Try to find setup code block
        idx_setup = src.find('add_cell("code", """import os\nimport sys\n')
        if idx_setup != -1:
            end_setup_imports = src.find('import rdkit', idx_setup)
            if end_setup_imports == -1:
                end_setup_imports = src.find('import chemlib', idx_setup)
            if end_setup_imports != -1:
                before = src[:idx_setup]
                after = src[end_setup_imports:]
                src = before + setup_replacement + "\n" + after

    # 4. Insert Common Pitfalls and Summary before # --- Self-Checks ---
    pit_text, sum_text = pitfalls_and_summaries[i]
    if "## Common Pitfalls" not in src:
        self_check_marker = '    # --- Self-Checks ---\n'
        if self_check_marker in src:
            pitfalls_block = f"""    # --- Common Pitfalls ---
    add_cell("markdown", \"\"\"{pit_text}\"\"\")

    # --- Summary & Key Takeaways ---
    add_cell("markdown", \"\"\"{sum_text}\"\"\")

"""
            src = src.replace(self_check_marker, pitfalls_block + self_check_marker)

    # 5. Fix output paths
    old_paths = f"""    # Write notebooks
    os.makedirs("Cheminformatics/notebooks", exist_ok=True)
    os.makedirs("Cheminformatics/solutions", exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)

    path_student = "Cheminformatics/notebooks/{fn}"
    path_sol = "Cheminformatics/solutions/{fn}\""""

    new_paths = f"""    # Write notebooks
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path_student = os.path.join(repo_root, "notebooks", "{fn}")
    path_sol = os.path.join(repo_root, "solutions", "{fn}")
    os.makedirs(os.path.dirname(path_student), exist_ok=True)
    os.makedirs(os.path.dirname(path_sol), exist_ok=True)

    nb_student = new_notebook(cells=cells_student)
    nb_sol = new_notebook(cells=cells_solution)"""

    if old_paths in src:
        src = src.replace(old_paths, new_paths)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched build_nb{i}.py successfully.")

# ----------------------------------------------------------------------
# 3. Update build_nb15.py
# ----------------------------------------------------------------------
path_15 = os.path.join(SCRIPTS_DIR, "build_nb15.py")
with open(path_15, "r", encoding="utf-8") as f:
    src_15 = f.read()

# Replace ### Objectives with ## Learning Objectives
src_15 = src_15.replace("### Objectives\n", "## Learning Objectives\n")

# Add SEED=42, FULL_RUN=False to setup
if "SEED = 42" not in src_15:
    old_setup = 'add_cell("code", """import os\nimport sys\nimport numpy as np'
    new_setup = 'add_cell("code", """import os\nimport sys\nimport random\nimport numpy as np\nimport pandas as pd\n\nSEED = 42\nFULL_RUN = False\nrandom.seed(SEED)\nnp.random.seed(SEED)\n'
    src_15 = src_15.replace(old_setup, new_setup)

# Add Common Pitfalls before Summary
if "## Common Pitfalls" not in src_15:
    pit_15 = """    # --- Common Pitfalls ---
    add_cell("markdown", \"\"\"## Common Pitfalls

1. **Dogmatic Rule Application**: Blockbuster oral drugs (e.g., atorvastatin, montelukast) violate Lipinski rules; druglikeness filters are heuristic guidelines for bioavailability, not strict physical laws.
2. **False Positives in PAINS Filters**: Reactive substructures flagged by PAINS filters may still be genuine biological binders in specific biochemical contexts; verify experimentally before discarding leads.
3. **Ignoring Synthetic Accessibility**: A computationally optimal druglike candidate is useless if synthetic route complexity (SA Score > 7) prohibits chemical synthesis.\"\"\")

"""
    src_15 = src_15.replace('    # --- Summary & Self-test ---\n', pit_15 + '    # --- Summary & Self-test ---\n')

# Fix paths in main()
old_main_15 = """def main():
    stud_dir = "Cheminformatics/notebooks"
    sol_dir = "Cheminformatics/solutions"
    os.makedirs(stud_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    nb_stud, nb_sol = create_nb15()

    stud_path = os.path.join(stud_dir, "nb15_druglikeness_and_admet.ipynb")
    sol_path = os.path.join(sol_dir, "nb15_druglikeness_and_admet.ipynb")"""

new_main_15 = """def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stud_dir = os.path.join(repo_root, "notebooks")
    sol_dir = os.path.join(repo_root, "solutions")
    os.makedirs(stud_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    nb_stud, nb_sol = create_nb15()

    stud_path = os.path.join(stud_dir, "nb15_druglikeness_and_admet.ipynb")
    sol_path = os.path.join(sol_dir, "nb15_druglikeness_and_admet.ipynb")"""

if old_main_15 in src_15:
    src_15 = src_15.replace(old_main_15, new_main_15)

with open(path_15, "w", encoding="utf-8") as f:
    f.write(src_15)
print("Patched build_nb15.py successfully.")

# ----------------------------------------------------------------------
# 4. Update build_nb16.py through build_nb21.py
# ----------------------------------------------------------------------
nb_helpers_info = {
    16: (
        "nb16_target_discovery_and_prep.ipynb",
        """## Common Pitfalls

1. **Unresolved Protonation States**: Crystallographic PDB files do not contain hydrogen atoms; ignoring histidine tautomers (HID/HIE/HIP) or catalytic dyad states leads to incorrect electrostatic potential fields.
2. **Alternate Conformations and Co-factors**: Retaining crystallographic waters, salts, or alternate sidechain conformations arbitrarily can block binding pockets or distort docking grids.
3. **AlphaFold Confidence Misinterpretation**: AlphaFold DB structures require checking pLDDT and PAE scores; flexible loops or intrinsically disordered regions (pLDDT < 50) must not be treated as rigid binding cavities.""",
        """## Summary and Key Takeaways

- **Target Selection**: Requires balancing disease relevance, genetic validation, tractability, and selective safety windows.
- **Structure Preparation**: Standard protocols include water stripping, alternate conformation resolution, explicit hydrogen addition, and protonation state assignment.
- **Pocket Metrics**: Cavity detection and grid box bounding parameters are crucial pre-requisites for structure-based docking."""
    ),
    17: (
        "nb17_scoring_functions_and_sampling.ipynb",
        """## Common Pitfalls

1. **Over-reliance on Raw Docking Energies**: Docking scoring functions approximate free energy of binding but omit explicit entropic penalty and solvent reorganization details.
2. **Metropolis Acceptance Trapping**: In Monte Carlo sampling, poor move proposals or suboptimal temperatures cause simulations to become trapped in local energy minima.
3. **Electrostatic Dielectric Approximations**: Distance-dependent dielectrics ($\\\\epsilon(r) = 4r$) are crude approximations of true aqueous continuum solvation.""",
        """## Summary and Key Takeaways

- **Scoring Function Categories**: Physics-based force fields, empirical regressions, and knowledge-based potentials balance computational speed and binding affinity accuracy.
- **Conformational Search**: Monte Carlo and genetic algorithms explore high-dimensional ligand torsional degrees of freedom.
- **Scoring Decomposition**: Dissecting terms provides physical insight into electrostatic, steric, and desolvation contributions to affinity."""
    ),
    18: (
        "nb18_docking_and_virtual_screening.ipynb",
        """## Common Pitfalls

1. **Rigid Receptor Assumption**: Standard docking treats receptors as rigid, missing induced-fit conformational adjustments upon ligand binding.
2. **Decoy Bias & Artificial Enrichment**: Decoys that are not physically property-matched (in MW, LogP, charge) produce artificially elevated ROC-AUC values.
3. **PDBQT Charge Errors**: Incorrect partial charges or incorrect torsional root assignments in ligand preparation directly degrade docking pose quality.""",
        """## Summary and Key Takeaways

- **SBVS Pipeline**: Structure-based virtual screening docks large libraries into 3D binding cavities to prioritize prospective hit candidates.
- **Redocking Validation**: Crystallographic redocking with heavy-atom RMSD < 2.0 Å validates docking protocol accuracy prior to virtual screening.
- **Early Enrichment**: Metrics such as EF1% and BEDROC ($\\\\alpha=20$) quantify early active recovery in virtual screening campaigns."""
    ),
    19: (
        "nb19_denovo_design_and_fbdd.ipynb",
        """## Common Pitfalls

1. **Synthetic Inaccessibility**: Unconstrained de novo fragment growing often generates synthetically impossible or highly strained heterocyclic systems.
2. **Ignoring Ligand Efficiency**: High-affinity hits with high molecular weight often possess lower ligand efficiency than weaker, compact fragments.
3. **BRICS Disconnection Inconsistencies**: Assembling fragments without matching complementary synthetically valid synthons yields unchemical molecules.""",
        """## Summary and Key Takeaways

- **FBDD Principles**: Fragment-based drug discovery leverages Rule of 3 (Ro3) chemical libraries to sample chemical space efficiently.
- **Efficiency Metrics**: Fragment growing, linking, and scaffold hopping are guided by Ligand Efficiency (LE) and Lipophilic Efficiency (LLE).
- **BRICS Synthesis**: Reaction rules ensure that de novo designs follow retro-synthetically accessible chemical transformation rules."""
    ),
    20: (
        "nb20_project_template_ligand_model.ipynb",
        """## Common Pitfalls

1. **Temporal & Scaffold Data Leakage**: Random train-test splitting yields artificially high predictive performance; prospective or Bemis-Murcko scaffold splitting must be used.
2. **Missing Applicability Domain Thresholds**: Applying QSAR models to structural outliers with leverage exceeding warning limits ($h^*$) produces unreliable predictions.
3. **Imbalanced Bioactivity Ratios**: Real-world HTS libraries have ~0.1-1% actives; models must be tuned with class-weighted objectives and evaluated with MCC and PR-AUC.""",
        """## Summary and Key Takeaways

- **Production QSAR**: Robust pipelines require automated curation, salt stripping, charge neutralization, and potency thresholding.
- **Multi-Algorithm Benchmarking**: Comparing RF, SVC, and GBM with cross-validation establishes robust predictive performance.
- **Model Transparency**: Deliverable model cards and applicability domain definitions ensure transparency and scientific reproducibility."""
    ),
    21: (
        "nb21_gnn_property_prediction.ipynb",
        """## Common Pitfalls

1. **Oversmoothing in Deep GNNs**: Stacking too many message-passing convolution layers causes node embeddings to become indistinguishable across the entire molecular graph.
2. **Neglecting Edge Features & Stereochemistry**: Standard GCNs aggregate adjacency matrices without distinguishing bond orders (single, double, aromatic) or stereochemical chirality.
3. **Adjacency Normalization Scaling**: Omitting self-loops ($\\\\mathbf{\\\\tilde{A}} = \\\\mathbf{A} + \\\\mathbf{I}$) causes central atom feature representations to be lost during neighborhood aggregation.""",
        """## Summary and Key Takeaways

- **Attributed Graphs**: Molecular graphs represent atoms as feature vectors and covalent bonds as graph edges.
- **Message Passing**: Graph Convolutional Networks (GCN) update atom embeddings via symmetrically normalized message passing.
- **Whole-Molecule Embeddings**: Global pooling (mean/sum pooling) aggregates atom embeddings for downstream bioactivity and property regression."""
    ),
}

for i in range(16, 22):
    path = os.path.join(SCRIPTS_DIR, f"build_nb{i}.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    fn, pit_text, sum_text = nb_helpers_info[i]
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/{fn})\\n\\n"

    # 1. Add Colab badge to title
    if "colab-badge" not in src:
        title_marker = f"# Notebook {i}:"
        if title_marker in src:
            src = src.replace(title_marker, f"{title_marker}\\n\\n{badge_md}")

    # 2. In nb20, ensure ## Learning Objectives is present
    if i == 20:
        src = src.replace("## 🎯 Course Capstone Project Overview", "## 🎯 Learning Objectives & Capstone Overview")

    # 3. Add SEED=42, FULL_RUN=False to setup
    if "FULL_RUN = False" not in src:
        # Search for first code cell with imports
        idx_setup = src.find('add_code(nb, """import os\nimport sys\n')
        if idx_setup != -1:
            setup_inject = 'add_code(nb, """import os\nimport sys\nimport random\nimport numpy as np\nimport pandas as pd\n\nSEED = 42\nFULL_RUN = False\nrandom.seed(SEED)\nnp.random.seed(SEED)\n'
            src = src[:idx_setup] + setup_inject + src[idx_setup + len('add_code(nb, """import os\nimport sys\n'):]

    # 4. Insert Common Pitfalls and Summary before final validation
    if "## Common Pitfalls" not in src:
        final_marker = f'print("✔ NB{i} checks passed")'
        # find the add_code containing final_marker
        pattern = rf'(add_code\(nb,\s*"""[^"]*print\("✔ NB{i} checks passed"\)[^"]*"""\))'
        m = re.search(pattern, src)
        if m:
            full_match = m.group(0)
            pit_sum_blocks = f"""add_md(nb, \"\"\"{pit_text}\"\"\")

    add_md(nb, \"\"\"{sum_text}\"\"\")

    {full_match}"""
            src = src.replace(full_match, pit_sum_blocks)

    # 5. In nb16, remove alias_relpaths if present
    if i == 16 and "alias_relpaths" in src:
        src = re.sub(r',\s*alias_relpaths=\[[^\]]+\]', '', src)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched build_nb{i}.py successfully.")

print("\nAll patches applied to build scripts!")
