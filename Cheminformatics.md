# Cheminformatics: Hands-on Course in Jupyter Notebooks

> **Master specification and layout document.**
> Source material: 8 lecture decks (IIIT Delhi) covering course introduction, structure representation, cheminformatics tools, organic structures, molecular descriptors, QSAR, similarity search and virtual screening.
> Purpose: (1) describe the repository to students, and (2) give an AI agent or contributor everything needed to build each notebook, dataset, exercise and test without re-reading the slides.

---

## Table of contents

1. [Agent brief: how to use this document](#1-agent-brief-how-to-use-this-document)
2. [Course overview, credits and learning outcomes](#2-course-overview-credits-and-learning-outcomes)
3. [Repository layout](#3-repository-layout)
4. [Environment and tooling](#4-environment-and-tooling)
5. [Global notebook conventions](#5-global-notebook-conventions)
6. [Course map (lecture → notebook)](#6-course-map-lecture--notebook)
7. [Notebook specifications](#7-notebook-specifications)
8. [Assignments, project and assessment](#8-assignments-project-and-assessment)
9. [Dataset registry](#9-dataset-registry)
10. [Tool registry (slide tool → Python equivalent)](#10-tool-registry-slide-tool--python-equivalent)
11. [Slide errata and modernization notes](#11-slide-errata-and-modernization-notes)
12. [References](#12-references)
13. [Definition of done and agent prompt template](#13-definition-of-done-and-agent-prompt-template)

---

## 1. Agent brief: how to use this document

**You are building a teaching repository.** Every notebook listed in [Section 7](#7-notebook-specifications) must be implemented as a runnable, self-explanatory `.ipynb` that follows [Section 5](#5-global-notebook-conventions).

Rules for the agent:

1. **Build in order** `nb00 → nb21`. Later notebooks import helpers written for earlier ones (`src/chemlib/`).
2. **One notebook = one spec block.** Do not add or drop sections without recording the change in `docs/CHANGELOG.md`.
3. **Respect the errata.** [Section 11](#11-slide-errata-and-modernization-notes) lists places where the slides are outdated, inconsistent or incorrect. Implement the *corrected* behavior and add a short "Note vs. slides" markdown cell wherever it matters.
4. **Never trust a tool blindly.** The slides explicitly teach that RDKit and Open Babel can give different results for the same SMILES (stereochemistry, aromaticity, 3D coordinates). Notebooks should *verify* outputs (round-trip, canonical SMILES comparison, sanity assertions) rather than assume them.
5. **CPU-only, laptop-friendly.** Every notebook must run top-to-bottom in under ~5 minutes (target) on a 4-core laptop without a GPU. Heavy steps (docking of many ligands, GNN training, PaDEL on thousands of molecules) must use small subsets with a documented switch (`FULL_RUN = False`) to scale up.
6. **Deterministic.** Fix random seeds (`SEED = 42`), pin library versions, and cache downloaded data under `data/`.
7. **No copyrighted slide reproduction.** Re-create figures programmatically (RDKit drawings, matplotlib). Cite the original papers named in the slides instead of embedding slide images.
8. **Every notebook ships with:** learning objectives, a prerequisites cell, theory recap (short), worked examples, ≥3 exercises with hidden-solution notebooks in `solutions/`, a "Common pitfalls" cell, a summary, and references.
9. **Flag uncertainty.** If a web resource named in the slides (e.g. a web server) is unavailable or changed, implement an offline alternative and say so in the notebook.

---

## 2. Course overview, credits and learning outcomes

### 2.1 What the course is

Cheminformatics (also spelled chemoinformatics; the term dates from 1998) is the design, creation, organization, storage, management, retrieval, analysis, dissemination, visualization and use of chemical information. The course walks from *how molecules are represented in a computer* to *how they are searched, described, modeled and screened*, framed around drug discovery as a multi-objective optimization problem (binding affinity and specificity first, then ADME/T, solubility and bioavailability).

### 2.2 Credits (from the course introduction deck)

| Component | Weight |
|---|---|
| 2 assignments | 30 |
| 1 project | 20 |
| Mid-term test | 20 |
| End test | 30 |

- **Assignment topics:** (A1) searching chemical libraries with a chemical template or pharmacophore; (A2) de novo design of molecules for a biological target (LigBuilder was suggested).
- **Project:** develop a ligand-based machine-learning model to identify inhibitors of a biomolecular target.

### 2.3 Learning outcomes

By the end, a student can:

- **LO1** Represent molecules as SMILES, InChI, molecular graphs, connection tables and 3D coordinates (Cartesian and Z-matrix), and convert between formats safely.
- **LO2** Query public chemical/bioactivity databases and curate a dataset (standardization, de-duplication, activity labeling).
- **LO3** Compute and interpret molecular descriptors (0D–4D) and fingerprints (key-based, path-based, circular, pharmacophore).
- **LO4** Perform substructure and similarity searches and explain the screening → verification pipeline.
- **LO5** Build, validate and critique QSAR/QSPR models (MLR and ML classifiers/regressors) with correct metrics.
- **LO6** Build a pharmacophore model and use it to search a library.
- **LO7** Prepare a target, dock ligands, validate a docking protocol, and run a small virtual screen.
- **LO8** Apply drug-likeness/ADME filters and understand fragment-based and de novo design.

### 2.4 Prerequisites

Basic Python (functions, lists, dicts, pandas), introductory organic chemistry, high-school-level statistics. No prior RDKit experience assumed (`nb00` covers it).

---

## 3. Repository layout

```text
cheminformatics-course/
├── README.md                      # student-facing quick start (short)
├── cheminformatics.md             # THIS file: master spec for contributors/agents
├── LICENSE
├── environment.yml                # conda env (primary)
├── requirements.txt               # pip fallback (pinned)
├── Makefile                       # make env | test | lint | notebooks | clean-outputs
├── .github/
│   └── workflows/ci.yml           # execute notebooks headless, run tests, lint
├── notebooks/
│   ├── part_A_foundations/
│   │   ├── nb00_setup_python_rdkit_refresher.ipynb
│   │   ├── nb01_introduction_chemical_space.ipynb
│   │   ├── nb02_building_3d_structures.ipynb
│   │   ├── nb03_smiles_and_linear_notations.ipynb
│   │   ├── nb04_molecular_graphs.ipynb
│   │   └── nb05_file_formats_and_conversion.ipynb
│   ├── part_B_data_and_chemistry/
│   │   ├── nb06_chemical_databases_chembl_pubchem.ipynb
│   │   └── nb07_functional_groups_and_isomers.ipynb
│   ├── part_C_descriptors_fingerprints/
│   │   ├── nb08_molecular_descriptors.ipynb
│   │   └── nb09_fingerprints.ipynb
│   ├── part_D_search_and_pharmacophores/
│   │   ├── nb10_substructure_and_similarity_search.ipynb
│   │   └── nb11_pharmacophore_modeling.ipynb
│   ├── part_E_qsar_and_ml/
│   │   ├── nb12_qsar_foundations.ipynb
│   │   ├── nb13_ml_qsar_classification_regression.ipynb
│   │   ├── nb14_3d_qsar_comfa_style.ipynb          # advanced/optional
│   │   └── nb15_druglikeness_and_admet.ipynb
│   ├── part_F_structure_based/
│   │   ├── nb16_target_discovery_and_structure_prep.ipynb
│   │   ├── nb17_scoring_functions_and_sampling.ipynb
│   │   ├── nb18_docking_and_virtual_screening.ipynb
│   │   └── nb19_denovo_design_and_fbdd.ipynb
│   └── part_G_capstone/
│       ├── nb20_project_template_ligand_based_model.ipynb
│       └── nb21_gnn_property_prediction.ipynb      # advanced/optional
├── solutions/                      # mirrors notebooks/ ; solution notebooks (instructor)
├── assignments/
│   ├── A1_library_search_template_or_pharmacophore/   # brief, rubric, starter.ipynb
│   └── A2_denovo_design/                              # brief, rubric, starter.ipynb
├── project/
│   ├── brief.md
│   ├── rubric.md
│   └── report_template.md
├── src/chemlib/                    # small reusable helpers (pip install -e .)
│   ├── __init__.py
│   ├── io.py                       # read/write smi, sdf, mol2 (via obabel), pdbqt, csv
│   ├── standardize.py              # salt strip, neutralize, tautomer/canonicalize, dedupe
│   ├── viz.py                      # grid drawing, highlight substructure, bit-info plots
│   ├── graphs.py                   # adjacency/distance/bond matrices, topological indices
│   ├── descriptors.py              # RDKit/Mordred/PaDEL wrappers -> DataFrame
│   ├── fingerprints.py             # unified fingerprint API + similarity matrices
│   ├── metrics.py                  # classification + regression metrics, CV helpers
│   ├── pharm.py                    # feature extraction, Pharm2D signature helpers
│   └── docking.py                  # receptor/ligand prep + Vina runner + RMSD/enrichment
├── data/
│   ├── README.md                   # provenance + licence for every file
│   ├── raw/                        # untouched downloads (git-ignored if large; fetched by script)
│   ├── processed/                  # cleaned CSV/SDF used by notebooks
│   ├── external/                   # third-party sample files (PDB, ligands)
│   └── fetch_data.py               # reproducible downloader with checksums
├── assets/
│   └── images/                     # programmatically generated figures only
├── tests/
│   ├── test_chemlib_*.py
│   └── test_notebooks_execute.py   # nbclient/papermill smoke test per notebook
└── docs/
    ├── CHANGELOG.md
    ├── glossary.md
    ├── cheat_sheets/               # SMILES, SMARTS, RDKit, obabel CLI, metrics
    └── slides_index.md             # lecture → notebook mapping, links (no slide copies)
```

---

## 4. Environment and tooling

### 4.1 Core stack (pin versions at build time)

| Purpose | Package | Notes |
|---|---|---|
| Cheminformatics core | `rdkit` | primary toolkit |
| Format conversion, 3D gen | `openbabel` (CLI `obabel` + `openbabel` Python) | needed for MOL2/PDBQT; compare with RDKit |
| Data | `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels` | |
| Plotting | `matplotlib`, `seaborn`, `py3Dmol` | `py3Dmol` for 3D views in notebooks |
| Graphs | `networkx` | adjacency matrices, VF2 isomorphism demos |
| Descriptors | `mordred` (see note), `padelpy` (needs Java ≥ 8) | PaDEL via Java JAR |
| Database access | `chembl_webresource_client`, `pubchempy`, `requests` | cache every response |
| Docking | `vina` (AutoDock Vina Python), `meeko` | replaces legacy MGLTools for PDBQT prep |
| Protein handling | `biopython`, `pdbfixer`/`openmm` (optional), `prody` (optional) | |
| ML extras (optional) | `xgboost`, `torch`, `torch_geometric` | only in advanced notebooks, guarded imports |
| Notebook tooling | `jupyterlab`, `nbclient`, `papermill`, `nbstripout`, `pytest`, `ruff` | |

Notes for the builder:

- Verify at build time whether the original `mordred` package installs cleanly with the pinned NumPy/Python; if not, use the community-maintained fork (`mordredcommunity`) and document it.
- Checkmol (functional-group analysis) is a Pascal command-line tool compiled with Free Pascal (`fpc`). Provide an install script in `docs/` **and** an RDKit SMARTS-based fallback so notebooks work without it.
- MayaChemTools (Perl) and Dragon (commercial) are covered conceptually only; do not make notebooks depend on them.
- Web servers named in the slides (SwissADME, PreADMET, SwissSimilarity, ChemDes, CASTp, SwissModel) are covered as *guided web exercises* with screenshots-by-description plus an offline RDKit equivalent.

### 4.2 `environment.yml` skeleton

```yaml
name: cheminf
channels: [conda-forge]
dependencies:
  - python=3.11
  - rdkit
  - openbabel
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - statsmodels
  - matplotlib
  - seaborn
  - networkx
  - biopython
  - jupyterlab
  - nbclient
  - papermill
  - nbstripout
  - pytest
  - ruff
  - openjdk        # for PaDEL
  - pip
  - pip:
      - py3Dmol
      - padelpy
      - mordredcommunity
      - chembl_webresource_client
      - pubchempy
      - vina
      - meeko
```

### 4.3 CI expectations

- GitHub Actions: create env → `pytest` → execute every notebook with `FULL_RUN=False` and a 10-minute per-notebook timeout.
- Network-dependent cells must be wrapped (`try/except`) and fall back to cached files in `data/processed/` so CI stays green offline.
- `nbstripout` pre-commit hook: committed notebooks contain **no outputs** (except in `solutions/` if desired).

---

## 5. Global notebook conventions

### 5.1 Standard cell skeleton (every notebook)

1. **Title cell** – `# NBxx · Title`, one-line summary, estimated time, difficulty (●○○ / ●●○ / ●●●), Colab badge.
2. **Learning objectives** – 3 to 6 bullet verbs (explain, compute, compare, validate…).
3. **Prerequisites and links** – earlier notebooks, lecture deck name, key terms.
4. **Setup cell**

```python
import os, random, warnings
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, rdFingerprintGenerator, DataStructs

SEED = 42
FULL_RUN = False          # set True to run the heavy variants
random.seed(SEED); np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")   # re-enable in the "pitfalls" section when teaching error messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

5. **Theory recap** – ≤ 1 screen of markdown with formulas in LaTeX.
6. **Worked examples** – small, commented, each followed by a "What to notice" markdown cell.
7. **Exercises** – numbered `Exercise N.M`, each with a stub cell (`# YOUR CODE HERE`) and an `assert`-based self-check cell (no solution revealed).
8. **Common pitfalls** – real failure modes (invalid SMILES, implicit vs explicit H, stereo loss, bit collisions, data leakage).
9. **Summary / key takeaways** – 4–6 bullets.
10. **References and further reading**.

### 5.2 Style rules

- Use `rdFingerprintGenerator` (modern API); show the legacy call only inside a "Note vs. slides" cell.
- Prefer small, readable functions; any function used in ≥2 notebooks moves to `src/chemlib/`.
- Always display molecules with `Draw.MolsToGridImage` or `IPythonConsole`, with legends.
- Plots: titled, axes labeled with units, colorblind-safe palette, `figsize` ≤ (8, 5).
- Assertions as formative checks, e.g. `assert Chem.MolToSmiles(m) == expected`.
- Each notebook ends with a **self-test cell** that runs all assertions and prints `✔ NBxx checks passed`.
- Text language: English, concise, no marketing tone.

### 5.3 Naming

- Notebook files: `nbNN_snake_case_title.ipynb`.
- Data files: `<source>_<content>_<YYYYMMDD>.csv` (e.g. `chembl_ache_ic50_20250101.csv`).
- Figures generated by code go to `assets/images/nbNN_<name>.png`.

---

## 6. Course map (lecture → notebook)

| Lecture deck | Main topics | Notebooks |
|---|---|---|
| 01 Course introduction | Definitions, chemical universe, SBDD vs LBDD, building 3D structures, SMILES basics, drug discovery as optimization, scoring and ML overview | nb00, nb01, nb02, nb03 (+ previews of nb13, nb18) |
| 02 Structure representation | Databases, graphs, matrices, InChI, SMILES, MOL, 3D formats, Z-matrix, stereochemistry | nb03, nb04, nb05, nb06 |
| 03 Cheminformatics tools | Editors, visualizers, Open Babel, RDKit, descriptors tools, VS, docking, de novo, FBDD, ADME servers | nb05, nb08, nb15, nb18, nb19 |
| 04 Organic structures & properties | Isomers, functional groups, Hückel, Checkmol, FG fingerprints, pharmacophore concept | nb07, nb09, nb11 |
| 05 Molecular descriptors | Descriptor classes 0D–4D, topological indices, QM, geometrical, PaDEL/Mordred/etc. | nb08 |
| 06 QSAR | Hansch, Free-Wilson, logP additivity, protocol, 3D-QSAR, metrics | nb12, nb13, nb14 |
| 06 Similarity search | Substructure search, fingerprints, graph isomorphism, similarity coefficients, Tanimoto/Tversky/Dice | nb09, nb10 |
| 07 Virtual screening | Target discovery, homology modeling, binding sites, Lipinski, scoring, docking, validation, hands-on BACE-1 | nb15, nb16, nb17, nb18 |

---

## 7. Notebook specifications

Each block lists: **Slide source**, **Objectives**, **Libraries**, **Content plan** (cell-level), **Data**, **Exercises**, **Checks / pitfalls**.

---

### Part A · Foundations

#### nb00 · Setup, Python and RDKit refresher  ●○○ · ~30 min
- **Slide source:** none (onboarding); tools named in decks 01 and 03.
- **Objectives:** install and verify the environment; read/write molecules with RDKit; call Open Babel from Python and CLI.
- **Libraries:** `rdkit`, `openbabel`, `pandas`.
- **Content plan:**
  1. Environment verification cell (`rdkit.__version__`, `obabel -V`, Java check for PaDEL, Vina import).
  2. `Chem.MolFromSmiles`, `MolToSmiles`, `Draw`, atom/bond iteration, implicit vs explicit H (`AddHs`, `RemoveHs`).
  3. Reading a SMILES file and an SDF into a DataFrame (`PandasTools`).
  4. Open Babel quick tour: `obabel -L formats | wc -l` (the slide shows 183 formats; report the actual number), one conversion, one `--gen3d` call.
  5. Error handling: what `MolFromSmiles` returns for invalid input; sanitization flags.
- **Exercises:** parse 5 SMILES (1 invalid) and report which fail and why; count heavy atoms and bonds; list all atoms with their formal charge.
- **Pitfalls:** `None` molecules propagating into pandas; silent log suppression.

#### nb01 · Introduction: informatics, chemical space and the drug-discovery problem  ●○○ · ~45 min
- **Slide source:** 01 Course introduction (nomenclature, informatics, molecular universe, chemicals in daily life, SBDD vs LBDD, drug design as optimization, data-driven vs deterministic).
- **Objectives:** define cheminformatics; reason about chemical-space size; contrast SBDD and LBDD; frame drug design as an optimization of binding affinity, specificity and ADME/T.
- **Content plan:**
  1. Timeline/definition cells (term coined 1998; Greg Paris quote context). Paraphrase, do not copy slides.
  2. **Chemical-space size:** back-of-envelope estimate (~10^60 drug-like molecules under constraints such as MW ≤ 500, elements C/H/O/N/S, limited rings) and **isomer growth** table from alkane counts (C7H16 → 9 isomers … C30H62 → ~4.1×10^9). Compute small cases by enumeration with RDKit/networkx to validate (up to ~C10).
  3. SBDD vs LBDD decision flowchart (draw with matplotlib or Mermaid in markdown): target discovery → 3D structure → virtual screening → de novo (atom-by-atom vs fragment) versus bioactivity dataset → pharmacophore/QSAR → library screening.
  4. Drug design as optimization: table of objectives (affinity, specificity, A, D, M, E, T, solubility, bioavailability). Show why ADME/T neglect causes clinical failures (qualitative).
  5. Free-energy view: ΔG_bind = RT ln K_i. Interactive cell converting K_i ↔ ΔG at 298 K (nM, pM, fM) to explain "sub-nanomolar to nanomolar" for approved drugs.
  6. Deterministic vs data-driven approaches: short comparison table (cost, data needs, when each wins—e.g., BBB penetration, toxicity, DSSC power efficiency).
- **Data:** none (optional small ChEMBL snippet for later notebooks).
- **Exercises:** compute ΔG for K_i = 10 µM, 10 nM, 10 pM; enumerate alkane isomers up to C8 using a graph-based approach and compare to the known counts (9, 18 for C8? — verify with the slide's table: C8H18 = 13 skeletal-tree isomers; make the notebook state exactly what is being counted, e.g. constitutional isomers of acyclic alkanes).
- **Pitfalls:** log units (ln vs log10), temperature, standard-state conventions.

#### nb02 · Building 3D structures: Cartesian, internal coordinates and editors  ●●○ · ~60 min
- **Slide source:** 01 (building molecules by trigonometry, orientation dependence, Z-matrix, Exercise-1), 02 (Cartesian coordinates, Z-matrix, 1,2-dichloroethane).
- **Objectives:** construct methane from geometry; show that coordinates are orientation dependent while internal coordinates and distances are not; write and read a Z-matrix.
- **Libraries:** `numpy`, `rdkit`, `py3Dmol`.
- **Content plan:**
  1. Build methane from r(C–H) and ∠HCH (tetrahedral 109.47°) using the slide's parametrization (θ = ∠HCH/2): place C at origin; H at (r cosθ, ±r sinθ, 0) and (−r cosθ, 0, ±r sinθ). Compute and verify all H–C–H angles and C–H lengths; compare with the 0.63/0.89 style coordinates on the slide (note these correspond to a specific r).
  2. **Orientation dependence:** random rotation matrix applied to coordinates → coordinates change, distance matrix and angles unchanged (assert with `np.allclose`). Answer the slide question "Are properties coordinate dependent?" (No for intrinsic properties).
  3. **Z-matrix:** parse/write a small Z-matrix (methane, n-butane, 1,2-dichloroethane with bond length, bond angle, dihedral); convert Z-matrix → Cartesian with NeRF-style placement; explain why dihedral assignment is harder for non-planar molecules.
  4. 3D viewing with `py3Dmol`; build n-butane and scan the C–C–C–C dihedral to show gauche (±60°) and anti (180°) minima with MMFF (preview of conformers).
  5. Molecular editors overview table: Molden, Avogadro, Maestro, plus 2D editors (XDrawChem, ISIS Draw, ChemDraw, JME, Marvin Sketch, PubChem sketcher). No installation required; short "how to export SMILES/MOL" instructions.
- **Exercises:** Exercise-1 from the deck: *build methane using internal coordinates* (Z-matrix) and convert to Cartesian; write coordinates of C and H1 after rotating the molecule about the x-axis by 30°.
- **Pitfalls:** degrees vs radians, dihedral sign conventions, right- vs left-handed frames.

#### nb03 · SMILES and other linear notations  ●●○ · ~75 min
- **Slide source:** 01 (SMILES rules, bonds, branches, rings, aromatics, "problem in using softwares"), 02 (linear notation, InChI, SMILES, stereochemistry in SMILES).
- **Objectives:** read and write SMILES by hand; understand canonicalization; express stereochemistry (`@`, `@@`, `/`, `\`); compare SMILES and InChI/InChIKey; spot tool disagreements.
- **Content plan:**
  1. SMILES grammar tour with live parse: atoms (organic subset B, C, N, O, P, S, F, Cl, Br, I; bracket atoms such as `[Au]`, `[NH4+]`), bond symbols `- = # :`, branches `()`, ring-closure digits (including multi-digit `%10`), aromatic lowercase atoms, disconnected components `.`.
  2. Hand-drawing practice: the slide's **10 SMILES to draw** — `CCO`, `CC(=O)O`, `CC(=O)OCC.O`, `C=CCBr`, `C#N`, `CCN(CC)CC`, `C(C(C(=O)O)N)O`, `OC(=O)C(Br)(Cl)N`, `ClC(Br)(N)C(=O)O`, `O=C(O)C(N)(Br)C`. Render each with RDKit and name it. **Observation exercise:** items 8 and 9 are the *same* molecule written differently → show canonical SMILES equality; item 3 is multi-component.
  3. **Write-the-SMILES task** (six drawn molecules from the deck: a CF3-phenothiazine with a diethylaminomethyl group, a pyrazolone (antipyrine-type, e.g. `CC1=CC(=O)N(N1C)c1ccccc1`), a cyclohexyl-polyene alcohol, a peptidomimetic with histidine and cyclohexylmethyl, a chiral tricyclic indole carboxylic acid, and a dihydrobenzofuran ketone with a terminal alkyne). The agent must generate reference SMILES, verify by depiction, and store them in `solutions/`.
  4. Canonicalization: same molecule, many valid SMILES; randomized SMILES (`doRandom=True`) for augmentation; `MolToSmiles` options (isomeric on/off).
  5. **Stereochemistry:** tetrahedral centers (`@`/`@@`), double-bond geometry (`/`, `\`), CIP labels via `Chem.AssignStereochemistry`, enumerating stereoisomers (`EnumerateStereoisomers`); the slide's example (2R,3E,5E)-2-hydroxy-3,5-heptadienenitrile. Link to biology: thalidomide enantiomers, fenoterol uptake by OCT1/OCT2 (qualitative).
  6. **InChI / InChIKey:** five layers (connectivity, tautomerism, isotopes, stereochemistry, charge); generate and compare InChIKey to SMILES-based identity; show a case where two SMILES give identical InChI.
  7. **"Never take it for granted":** take the large steroid-like SMILES from the slide (`[C@@]12([C@H](C3=C(...)...`), parse in RDKit and Open Babel, compare canonical SMILES, stereocenter counts and InChI; report any differences.
- **Exercises:** hand-write SMILES for aspirin, caffeine, sertraline (stereo!); detect invalid SMILES in a list; implement a mini SMILES validator that reports the reason.
- **Pitfalls:** lowercase aromatic atoms in non-aromatic contexts, un-closed rings, lost stereo on `RemoveHs`, canonical SMILES differing across toolkits and versions.

#### nb04 · Molecular graphs and matrices  ●●○ · ~75 min
- **Slide source:** 02 (graph theory, subgraphs, adjacency/distance/bond matrices, advantages, isomorphism, GNN solvation example), 05 (molecular graph, walks, paths, distances), 06 similarity (subgraph isomorphism).
- **Objectives:** treat a molecule as a (colored) graph; build adjacency, distance and bond-order matrices; explain graph/subgraph isomorphism and its complexity.
- **Libraries:** `networkx`, `rdkit`, `numpy`.
- **Content plan:**
  1. Molecule → `networkx.Graph` with node attributes (element, aromatic, charge) and edge attributes (bond order). Show that drawings with different layouts are the same graph ("graphs are only about connectivity").
  2. Hydrogen-depleted graph vs full graph; vertex degree.
  3. Matrices for aspirin (and a small example): **adjacency**, **bond-order**, **topological distance** (Floyd–Warshall / `nx.floyd_warshall_numpy`) and **connection table** (MOL-block style atom/bond lists). Compare with `Chem.GetAdjacencyMatrix`, `Chem.GetDistanceMatrix`.
  4. **Walks, paths, cycles:** definitions, walk vs path counts, topological distance vs detour distance, self-returning walks (`trace(A^k)`); implement molecular walk count MWC_k and self-returning walk count SRW_k; verify MWC_0 = atoms, MWC_1 = bonds (as on the slides).
  5. **Isomorphism:** `nx.is_isomorphic`, `nx.algorithms.isomorphism.GraphMatcher` (VF2) with node/edge matching on element and bond order; subgraph test benzene ⊂ aspirin; compare with RDKit `HasSubstructMatch`. Timing study on growing random graphs.
  6. Complexity box: **subgraph isomorphism is NP-complete**; general graph isomorphism is not known to be NP-complete (see errata). Historical algorithms: Ullmann, Schmidt–Druffel (SD), Nauty (McKay), VF (Foggia 1999), VF2 (Foggia 2001); *recreate the benchmark idea* with Ullmann-like backtracking vs VF2 on synthetic graphs (small sizes).
  7. Optional teaser: one-layer message passing on the adjacency matrix (links to nb21).
- **Exercises:** implement `adjacency_from_smiles()`; prove that identical graphs ⇔ identical molecules fails when stereo/tautomers matter (demonstrate with two stereoisomers); count rings via cycle basis.
- **Pitfalls:** multigraph vs simple graph for bond orders; aromatic bond handling; symmetry causing multiple matches.

#### nb05 · File formats and conversion (2D ⇄ 3D)  ●●○ · ~75 min
- **Slide source:** 01 (SMILES → mol2 with obabel and RDKit, online converter), 02 (MOL, 3D formats: xyz, PDB, MOL2; "how many file formats"), 03 (Open Babel, 3D generation + MMFF94/GAFF/UFF, force-field form, RDKit pipeline).
- **Objectives:** read/write MOL/SDF, MOL2, PDB, XYZ, PDBQT; generate 3D from SMILES in both toolkits; compare coordinates and energies; add charges.
- **Content plan:**
  1. Format anatomy: print and annotate a MOL block (counts line, atom block, bond block), SDF multi-record and data fields, XYZ, PDB `ATOM/HETATM`, MOL2 sections (`@<TRIPOS>ATOM/BOND`, partial charges), PDBQT (docking input: coordinates + charges + atom types). Table: which formats hold 2D/3D, charges, bond orders, multiple molecules.
  2. **0D → 2D → 3D** interconversion for sertraline (stereo SMILES `CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21`-type; verify the exact stereo string against a trusted source at build time): `obabel -ismi -:"..." -omol2 -h --gen3d`, and RDKit `AddHs → EmbedMolecule → MMFFOptimizeMolecule`.
  3. **Force-field primer:** E = E_str + E_bend + E_torsion + E_nonbond; show the harmonic bond/angle, periodic torsion and Lennard-Jones/Coulomb forms as plotted functions. Compare UFF vs MMFF94 energies for the same conformer. Note the correct statement about non-bonded terms (see errata).
  4. **3D generation recipe in Open Babel** (rule-based fragment build by hybridization → stereo correction → MMFF94 minimization with weighted rotor search; GAFF/UFF alternatives) vs ETKDG in RDKit; conformer ensembles and RMSD between toolkits (aligned RMSD and unaligned coordinate difference).
  5. **Partial charges:** Gasteiger (RDKit), MMFF94 charges, Open Babel `--partialcharge`; show where charges are stored and which formats persist them.
  6. **Batch conversion:** convert a SMILES list to SDF, MOL, MOL2 (via `obabel`) and PDBQT (via Meeko); validate round-trip (atom count, formal charges, stereo).
  7. Pitfalls demonstration: the slide snippet writes a `.mol` file with `SDWriter` although titled "to Mol2"; show what is lost.
- **Exercises:** *Home assignment 1 items* — build ethyl chloride and store as `.mol2`; take a random SMILES, generate 3D coordinates with `obabel` and RDKit, and test whether coordinates are identical (they will not be: explain why); write a function `convert(smiles, fmt)`.
- **Pitfalls:** missing explicit H before embedding, random conformer seeds, MOL2 atom-type conventions, multi-fragment inputs.

---

### Part B · Data and organic chemistry

#### nb06 · Chemical databases: PubChem, ChEMBL, ZINC and friends  ●●○ · ~90 min
- **Slide source:** 02 (databases, CSD, PDB, ChEMBL), 04 (public domain libraries, contents of libraries), 03 (home assignment: ChEMBL AChE), 07 (small-molecule databases, ZINC properties).
- **Objectives:** know what each database contains; retrieve bioactivity data programmatically; standardize and curate a dataset; produce a clean CSV of SMILES + activity.
- **Libraries:** `chembl_webresource_client`, `pubchempy`, `pandas`, `rdkit`.
- **Content plan:**
  1. **Database atlas** (table): PubChem, ChEMBL, DrugBank, eDrugs, ZINC (incl. fragment subsets), Enamine, YMDB/HMDB, FlavorDB, BindingDB, PDBBind, Cambridge Structural Database (>400k organic/organometallic X-ray structures per slide), Protein Data Bank. Columns: content, size order of magnitude (state as "as of build date" and verify), access method, licence.
  2. Look inside files: the slide shows a SDF head and a YMDB `.smi/.csv` head — recreate by printing the first records of a small bundled SDF and CSV; explain fields (name, SMILES with explicit H, InChIKey).
  3. **ChEMBL hands-on (home assignment):** query *human acetylcholinesterase* (resolve target ChEMBL ID programmatically, do not hard-code unverified IDs), filter bioactivity type (IC50/Ki), units (nM), relation (`=`), assay confidence; write raw CSV; extract `canonical_smiles` + `standard_value` into a second CSV; convert pIC50 = −log10(IC50 [M]).
  4. **Curation pipeline** (`chemlib.standardize`): remove salts/solvents, neutralize, canonical SMILES, drop invalid, handle duplicates (median/aggregate with IQR flag), outlier checks, activity thresholding for classification (e.g. active ≤ 1 µM, inactive ≥ 10 µM; make thresholds parameters).
  5. **Structure-data mining idea (CSD/PDB):** use a small bundled torsion dataset (or RDKit conformer scan) to reproduce the slide insight that O–C–C–O prefers gauche (±60°) while C–C–C–C prefers anti; ester carbonyl O is a better H-bond acceptor than the ether O. Present as optional, with data provided in `data/external/`.
  6. **Property distributions** of a library subset (MW, HBD, HBA, cLogP, rotatable bonds, Lipinski violations) — preview of nb15.
  7. Save outputs to `data/processed/` with a provenance JSON (query, date, ChEMBL release, row counts).
- **Exercises:** repeat for another target (BACE-1, MAO-B); compare PubChem vs ChEMBL coverage for 5 drugs; quantify duplicates before/after curation.
- **Pitfalls:** unit heterogeneity, mixed assay types, censored values (`>`, `<`), activity cliffs, data leakage via duplicates across splits.

#### nb07 · Organic structures, isomers and functional groups  ●●○ · ~75 min
- **Slide source:** 04 (organic vs inorganic, sp3 carbon, isomers, functional groups, aromaticity, Checkmol, functional-group fingerprints, home assignment).
- **Objectives:** recognize and encode functional groups (SMARTS); count isomers; apply Hückel's rule; build a functional-group fingerprint.
- **Content plan:**
  1. Organic chemical space: why carbon yields huge diversity (≥16 million organic vs ~0.6 million inorganic compounds per slide—state as historical figures); sp3 geometry (109.5°) check with 3D embedding of methane/ethane.
  2. **Isomers:** constitutional vs stereo; reproduce the alkane isomer growth table (C7 → 9 … C40 → 6.2×10^13) via a formula/recurrence or OEIS values; enumerate C6H14 isomers explicitly and draw.
  3. **Functional-group SMARTS library** (hydrocarbons, aromatics, haloalkanes, alcohol/phenol/thiol, ether/sulfide, amines (1°/2°/3°), aldehyde/ketone/imine, carboxylic acid and derivatives (ester, amide, anhydride, acyl halide), nitrile, organophosphates, etc.). Implement `functional_group_counts(mol)`; unit tests per group.
  4. **Aromaticity:** Hückel 4n+2; examples the slide mentions (1-methyl-2(1H)-pyridone, cyclopentadienyl anion, tropylium, fulvene, tropone) — compare RDKit's aromaticity perception with a hand Hückel check; discuss model differences.
  5. **Function → property:** boiling point of methane vs methanol (H-bonding), effect of single functional groups; additive binding-energy intuition for multi-FG inhibitors.
  6. **Checkmol:** (a) if installed, run on SDF/MOL (`checkmol file.mol`, fingerprint modes) and parse output; (b) SMARTS fallback; (c) compare agreement on a test set. Document the Free Pascal build in `docs/`.
  7. **Functional-group fingerprint:** binary vector (presence/absence; typical length hundreds to thousands of bits); similarity between molecules using these vectors (preview of nb10).
  8. **Case study preview:** P-glycoprotein substrate/inhibitor classification using ~250 functional-group bits (kNN, RF, SVM; dataset of 484 + 1,935 compounds in the cited study) — implemented with a public substitute dataset in nb13.
- **Exercises:** *Home assignment 1 items* — for a chosen molecule identify all functional groups and produce the FG fingerprint; reconcile RDKit-SMARTS and Checkmol results.
- **Pitfalls:** overlapping SMARTS (ester vs ketone vs carboxylic acid), tautomers, charged forms.

---

### Part C · Descriptors and fingerprints

#### nb08 · Molecular descriptors  ●●● · ~120 min
- **Slide source:** 05 Molecular descriptors, 06 QSAR (0D–3D descriptors, desirable attributes), 03 (tools).
- **Objectives:** define and classify descriptors; compute topological, geometrical and quantum-chemical descriptors; use multiple toolkits; clean a descriptor matrix.
- **Libraries:** `rdkit`, `mordred(community)`, `padelpy`, `numpy`, `pandas`, optional `xtb`/`psi4`/`pyscf` (guarded).
- **Content plan:**
  1. **Definition** (Todeschini & Consonni): descriptor = result of a logical/mathematical procedure transforming chemical information into a number. Taxonomies: experimental vs calculated; global vs local vs field; counting/empirical/fragment/graph/topological; **0D, 1D, 2D, 3D, 4D** (with 13 desirable attributes from the QSAR deck as a checklist).
  2. **0D/1D:** atom counts, MW, fragment/functional-group counts, hydrophilicity index (empirical; give the formula only after verifying it against the cited source at build time; slide gives range −1 < Hy < 3.64).
  3. **2D topological:** vertex degree δ_i; total adjacency index (= 2 × bonds); **Zagreb indices** M1 = Σδ_i², M2 = Σδ_iδ_j over bonds; **Randić** connectivity index and its correlation with boiling points of alkanes (build a tiny alkane dataset with literature b.p.); **Wiener index** W = ½ΣD_ij; molecular walk counts (from nb04). Implement all from scratch with numpy and verify against RDKit/Mordred where available.
  4. **3D geometrical:** 3D-Wiener index, radius of gyration (mass-weighted formula), PMI-based shape (NPR1/NPR2), molecular volume/surface (`rdFreeSASA` or `rdMolDescriptors`); stress that these need a consistent conformer.
  5. **4D:** conformer ensemble averaging (descriptor distributions over 50 ETKDG conformers).
  6. **Quantum-chemical descriptors:** HOMO, LUMO (**lowest *unoccupied*** MO), gap, dipole moment, polarizability α = (αxx+αyy+αzz)/3, EA, IP, heat of formation, electrostatic potential. Implement a lightweight path (semi-empirical via xtb *if installed* or Psi4/PySCF for ≤ 15-atom molecules), otherwise load precomputed values. Reproduce the idea of the toxicity model (IGC50 vs logP and E_LUMO) on a small provided dataset: fit, report n, r², q²(LOO), s.
  7. **Tool bake-off:** RDKit (~200 descriptors), Mordred (1,600+), PaDEL (797 = 663 1D/2D + 134 3D; plus fingerprint types), MayaChemTools (conceptual), ChemDes web server (thousands of descriptors; check availability), Dragon (commercial, conceptual). Run PaDEL both ways: Python wrapper and the CLI pattern from the slides  
     `java -jar PaDEL-Descriptor.jar -maxruntime -1 -2d -3d -fingerprints -dir <smiles_or_mol_dir> -file out.csv -descriptortypes PaDEL_Descriptors.xml -usefilenameasmolname -retainorder` (verify flag spelling against the installed version's help text) and show editing `PaDel_Descriptors.xml` to enable only the PubChem fingerprinter.
  8. **Descriptor hygiene:** remove constant/near-constant, handle NaN/inf, correlation filter (|r| > 0.95), scaling, outlier check; why "independence" and "not trivially related to other descriptors" matter.
  9. **Cost vs information:** timing benchmark 0D → 2D → 3D on 200 molecules.
- **Data:** small alkane b.p. set; 100–300-molecule subset of the curated ChEMBL set from nb06; toxicity-like toy set with provenance.
- **Exercises:** compute Wiener/Randić for all C5–C7 alkane isomers and correlate with b.p.; compare ten RDKit descriptors against Mordred equivalents; select 10 non-redundant descriptors by variance + correlation filter.
- **Pitfalls:** 3D descriptors computed on arbitrary conformers, units, descriptors that fail silently (NaN), leakage when scaling before the split.

#### nb09 · Fingerprints: keys, paths, circles and pharmacophore pairs  ●●○ · ~90 min
- **Slide source:** 06 Similarity search (fingerprint taxonomy), 04 (FG fingerprints, Pharmacoprint), 03 (descriptor tools).
- **Objectives:** compute and compare the four fingerprint families; understand hashing, folding and bit collisions; choose a fingerprint for a task.
- **Content plan:**
  1. **Substructure-key fingerprints:** MACCS (166 keys; SMARTS-defined), PubChem (881 bits; via PaDEL or Open Babel-compatible implementation), Klekota–Roth (optional).
  2. **Path-based:** enumerate linear/branched paths up to length *k*, hash to a fixed address space; RDKit topological (Daylight-like, 2048 bits) and Open Babel FP2 (1024 bits, fragments of 1–7 atoms). Show that distinct paths (e.g. N–Cα–C vs N–Cα–Cβ) can **collide**.
  3. **Circular:** Morgan/ECFP (radius 2 → ECFP4, radius 3 → ECFP6), FCFP (feature invariants), count vs bit vectors, folding length effect, `bitInfo` to map bits back to atom environments and draw them (reproduce the "radius = 1/2 around each atom" idea).
  4. **MFP chemical features:** donor, acceptor, aromatic, halogen, basic, acidic definitions (SMARTS shown in the deck; use RDKit's `BaseFeatures.fdef` and show the differences).
  5. **2D pharmacophore fingerprints:** feature points → inter-feature topological distances → bin → bit ID (example in slides: HAcc–Acid–Arom triangle with distances 4/4/4). Use `rdkit.Chem.Pharm2D` (Gobbi) and custom feature definitions; show signature size arithmetic from the deck (2 patterns, 2–3-point pharmacophores, 2 distance bins ⇒ 6 + 32 = 38 bits) and how Pharmacoprint reaches 39,973 bits (8 features, 7 distance bins, all 2- and 3-point pharmacophores) — reproduce the *counting*, not the proprietary implementation. Also ERG (extended reduced graph) and Estate fingerprints.
  6. **Summary table of fingerprints** (FP2, Daylight-like, MACCS, EState, ECFP4, Pharm2D, ERG: description, length, package) — regenerate programmatically by computing real lengths (note the FP2 length discrepancy in errata).
  7. Collision and information experiments: fold ECFP to 256/512/1024/2048/4096 bits and measure Tanimoto distortion; sparsity statistics.
  8. Storage/speed: `ExplicitBitVect` vs `SparseBitVect` vs numpy arrays; bulk similarity benchmark.
- **Exercises:** produce an ECFP4 bit-to-substructure gallery for aspirin; build a custom 3-point pharmacophore key; choose the best fingerprint to retrieve known analogs (preview of nb10 evaluation).
- **Pitfalls:** comparing fingerprints of different lengths/types, forgetting to sanitize, using counts with bit metrics.

---

### Part D · Searching and pharmacophores

#### nb10 · Substructure and similarity search  ●●● · ~120 min
- **Slide source:** 06 Similarity search (rationale, approaches, substructure search via fingerprints / graph theory / hybrid, similarity coefficients, Tanimoto, Tversky, Dice, SwissSimilarity, deep-learning VS), 03.
- **Objectives:** explain why similar structures tend to have similar activity; implement a two-stage substructure search; compute and compare similarity coefficients; run a ligand-based virtual screen and evaluate it.
- **Content plan:**
  1. Motivation: similar-property principle; expanding a known active set; chemistry beyond drugs (molecular probes for fibrils/RNA/DNA/G-quadruplexes; DSSC dyes with band-gap and optical properties).
  2. **Three approaches:** substructure, pharmacophore-based, similarity-index-based (LBVS).
  3. **Substructure search, toy then real:** the dopamine-like (catechol-ethylamine) query against adrenaline, mefeclorazine, olmidine, fenoldopam, apomorphine, morphine (deck example) — write SMILES, run `HasSubstructMatch`, highlight matches.
  4. **Hybrid screening → verification:** implement a pattern-fingerprint pre-filter (bit-subset test: query bits ⊆ target bits) followed by VF2 verification; measure speedup and false-positive rate on 50k molecules (subset). Recreate the deck's 5-bit toy example (query vs molecules A and B) as a visual.
  5. **Graph-theoretic view:** link to nb04 (VF2 timings; Ullmann; NP-completeness caveat).
  6. **Similarity coefficients:** define the 2×2 contingency (a = shared, b = x only, c = y only, d = neither; p = a+b+c+d). Implement **simple matching, Jaccard–Tanimoto, Dice/Sørensen–Gleason, Yule Q, cosine/Ochiai, Kulczynski, Sokal–Sneath, Russel–Rao, Simpson, Braun–Blanquet, Forbes** … (the deck lists 51; implement ≥ 15 as a vectorized dictionary). Reproduce the deck's numeric example (x = 1101100011, y = 1010100011 → a=4, b=2, c=1, d=3, p=10 — note the printed example in the slide table differs slightly from the vectors; the notebook should compute from the vectors and state the result).
  7. **Tanimoto in depth:** set-based definition T = N_c/(N_a + N_b − N_c); reproduce the 6-bit/6-bit/2-common = 0.20 figure; real examples with ECFP4 and PubChem fingerprints; threshold folklore (e.g. 0.85 for ECFP4 vs other fingerprints) — show empirical distributions of random pairs vs analog pairs instead of asserting a universal cut-off.
  8. **Tversky and Dice:** Tversky = c/(αa + βb + c) with a, b the *unique* bit counts; α = β = 1 ⇒ Tanimoto; α = β = 0.5 ⇒ Dice; asymmetric α ≫ β for substructure-like searches. (Note the notation clash with the coefficient slide in the errata.)
  9. **LBVS benchmark:** pick actives (e.g. AChE from nb06) + decoys (property-matched, DUD-E-style or random ChEMBL inactives); rank by max-similarity to query set; evaluate with ROC-AUC, PR-AUC, enrichment factor EF1%, BEDROC; compare fingerprints × coefficients in a heat map.
  10. **Scaffold-based and "scaffold hopping" search:** Bemis–Murcko scaffolds, generic scaffolds (the SwissSimilarity method list: FP2, ECFP4, MHFP6, pharmacophore, ErG, scaffold, generic scaffold, electroshape, E3FP — document which are reproducible offline).
  11. **SwissSimilarity guided exercise:** submit propranolol (the deck's example result lists naphthyloxy-propanolamine analogs with scores 1.000, 0.525, 0.510…) to the web tool, then reproduce with local ChEMBL/LigandExpo-like subsets and compare rankings. Fully optional if offline.
  12. **Deep-learning VS teaser:** 2D pharmacophore fingerprints → feature selection → DNN or reshaped-matrix CNN (concept from the deck; implement the DNN branch with scikit-learn `MLPClassifier`; CNN optional).
- **Exercises:** *Assignment A1 warm-up* — search a 20k-compound library with (i) a SMARTS template and (ii) a 3-point pharmacophore; report hit counts, enrichment, and 10 diverse hits.
- **Pitfalls:** Tanimoto bias towards small molecules, activity cliffs, comparing scores across fingerprints, tautomer/stereo handling in queries.

#### nb11 · Pharmacophore modeling and pharmacophore-based search  ●●● · ~120 min
- **Slide source:** 04 (pharmacophore definition, Fischer lock-and-key, early models: sulfonamide/PABA, estradiol/diethylstilbestrol, feature types, interaction types, feature matching and alignment, 5-HT2C antagonists, Pharmacoprint), 06 Similarity search (2D pharmacophore fp).
- **Objectives:** define a pharmacophore; extract features; build ligand-based 3D pharmacophores from aligned actives; search a library and score hits.
- **Content plan:**
  1. **Concept:** IUPAC-style definition (abstract ensemble of steric/electronic features necessary for optimal interaction with a target — *not* a real molecule or fixed set of functional groups); Fischer's lock-and-key; historical models (PABA vs sulfonamide critical distances; estradiol vs diethylstilbestrol).
  2. **Feature types and interactions:** hydrophobic, aromatic, HB donor/acceptor, cation, anion; vdW (hydrophobic–hydrophobic), electrostatics, H-bonding, π–π stacking. Feature extraction with RDKit `ChemicalFeatures` factory, customizing `.fdef`.
  3. **Common-feature pharmacophore from actives:** take a small active set (e.g. 5-HT2C antagonists or AChE inhibitors from ChEMBL; provide a curated 10–20-molecule set), generate conformers (ETKDG, 100 each), pairwise or multi-ligand alignment by feature matching (`Pharm3D` / O3A for shape+feature alignment), identify consensus features and inter-feature distances with tolerances; visualize with `py3Dmol` (HB acceptor = green, positive ionizable = red, hydrophobic = cyan, as in the deck's figure).
  4. **Scoring:** feature-match score, volume overlap, conformational strain energy (MMFF), selectivity (inactives/decoys rejection).
  5. **Pharmacophore search:** (a) 3D search over pre-generated conformers with a 3-point query; (b) 2D pharmacophore fingerprint search using Pharm2D (fast); compare hits, time and enrichment against nb10 baselines.
  6. **3-point pharmacophores explained:** triangle with three distances and feature labels; combinatorial counting of 3-point keys; distance binning (the deck's seven bins: <1–3, 3–4, 4–5, 5–6, 6–7, 7–8, 8–100) and how bin choice changes resolution.
  7. **Structure-based pharmacophore (optional):** derive features from a protein–ligand complex (H-bond donors/acceptors, hydrophobic contacts) using PLIP/ProLIF if available, else simple distance rules.
  8. **Limitations:** conformational flexibility, alignment ambiguity, dependence on training set.
- **Exercises:** *Assignment A1* — design a pharmacophore for a chosen target, search a provided ZINC subset, and justify the top 10 hits; compute enrichment on a held-out active/decoy set.
- **Pitfalls:** too many conformers (combinatorial explosion), mis-assigned protonation states for charged features, over-tight distance tolerances.

---

### Part E · QSAR, ML and ADME

#### nb12 · QSAR foundations: from Crum-Brown to Hansch  ●●○ · ~90 min
- **Slide source:** 06 QSAR (stages of molecular discovery, models, why QSAR, applications, data matrix, SAR vs QSAR, Crum-Brown & Fraser, Hansch, Free-Wilson, logP additivity, non-additivity, protocol, MLR metrics).
- **Objectives:** explain the QSAR workflow and rationale (cost, 3Rs, green chemistry); fit Hansch/Free-Wilson models; compute metrics for MLR; understand non-additive effects.
- **Content plan:**
  1. Design → synthesize → test → analyze → model loop (iterative optimization diagram). Why QSAR/QSPR (cost savings; Replace-Reduce-Refine of animal tests; early hazard identification) and application list (leads, optimization, toxicity, environmental fate, ADME, mixtures…).
  2. **Data matrix:** chemical identifier × activity × descriptors/fragments. SAR (qualitative yes/no) vs QSAR (quantitative).
  3. **Crum-Brown–Fraser:** Φ = f(C), ΔΦ ↔ ΔC; remark on historical context.
  4. **Hansch analysis:** linear model log(1/C) = k₁ log P + k₂σ + k₃ and parabolic model log(1/C) = −k₁(log P)² + k₂ log P + k₃σ + k₄; explain the optimum logP (membrane partitioning vs retention). **Simulate** a dataset with a known parabolic optimum, fit linear vs parabolic with `statsmodels`, show the fitted optimum logP* = k₂/(2k₁) and compare AIC/BIC.
  5. **Free–Wilson:** log 1/C = Σaᵢ + μ (substituent contributions + reference compound); implement with indicator variables on a small R-group series; discuss rank deficiency/reference choice. **Mixed Hansch/Free-Wilson** model (both functional forms shown in deck, including the bilinear form log P − b log(βP+1)).
  6. **logP additivity worked example:** m-chlorotoluene from π-constants: from benzene 2.13 + 0.71 + 0.56 = 3.40; from toluene 2.69 + 0.76 = 3.45; from chlorobenzene 2.84 + 0.51 = 3.35; experimental 3.28 → show the spread and compare with RDKit Crippen logP.
  7. **Non-additivity:** intramolecular H-bonds, ortho effects, polysubstituted aromatics, push–pull conjugation, heterocycles, cyclophanes (p-xylene vs [2.2]paracyclophane: surface areas 150.3 vs 196.6 and exp. logP 3.15 vs 2.33 as a case where size and logP trends diverge). Compute Crippen/XLogP-type estimates and discuss failures.
  8. **QSPR/QSAR protocol:** structure entry → descriptor calculation → descriptor selection → model building → validation. Desirable descriptor attributes (13-item checklist).
  9. **MLR metrics:** R², adjusted R², standard error s = √(Σ(Yobs−Ycalc)²/(N−p−1)), F-test, observed-vs-calculated plot with identity line; then **validation**: LOO q², external R²_pred, Y-randomization, applicability domain (leverage / Williams plot), and OECD principles (defined endpoint, unambiguous algorithm, domain, goodness-of-fit/robustness/predictivity, mechanistic interpretation).
- **Exercises:** fit Hansch models for a provided antibacterial/enzyme-inhibition series; do a Free–Wilson analysis on a 20-compound R-group table; run Y-randomization and interpret.
- **Pitfalls:** overfitting with too many descriptors per compound, extrapolating outside the domain, interpreting correlation as mechanism, chance correlations.

#### nb13 · Machine learning for QSAR (classification and regression)  ●●● · ~150 min
- **Slide source:** 01 (ML approaches, metrics, P-gp classification, MAO-B regression), 04 (metrics), 06 QSAR (classification metrics, MLR metrics), 05/03 (feature tools).
- **Objectives:** build a leakage-free ligand-based model; select appropriate metrics; compare algorithms; interpret and report results. **This notebook is the template for the course project.**
- **Libraries:** `scikit-learn`, `pandas`, `rdkit`, optional `xgboost`, `shap`.
- **Content plan:**
  1. **Problem framing:** dataset of positives (inhibitors) and negatives (non-inhibitors) or continuous pIC50/K_i; feature extraction (fingerprints, physchem descriptors, rule-of-five features); train/test split (deck: ~80/20) — extended with **scaffold split** and **time split** to reflect real generalization.
  2. **Baselines:** majority class, random, similarity-to-nearest-active.
  3. **Classifiers:** logistic regression, k-nearest neighbours (the deck says "kappa"-nearest—i.e. *k*-NN), random forest, SVM, gradient boosting, MLP. Hyper-parameter search with nested/stratified CV.
  4. **Regressors:** MLR/ridge/lasso, kNN-regressor, random forest, multilayer perceptron (as in the MAO-B case with ~1,054 PaDEL descriptors, normalization and feature selection).
  5. **Metrics:** confusion matrix (TP/FP/FN/TN); sensitivity (recall) = TP/(TP+FN); specificity = TN/(TN+FP); accuracy; precision; **F1 = 2·precision·recall/(precision+recall)** (see errata); balanced accuracy; **MCC** = (TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)); ROC-AUC, PR-AUC; calibration; for regression R², RMSE, MAE, q², Spearman. Explain false negatives (actives predicted inactive) vs false positives (inactives predicted active) in screening costs. Implement all metrics from scratch and compare with `sklearn.metrics`.
  6. **Case study A – P-glycoprotein:** use a public P-gp / ABCB1 ChEMBL or TDC-style dataset in place of the original 484 + 1,935 compounds; features: functional-group bits (nb07) vs ECFP vs physchem; compare kNN/RF/SVM like the cited WEKA study.
  7. **Case study B – MAO-B (or BACE-1) K_i regression:** PaDEL/Mordred descriptors → clean → normalize → feature selection (variance, correlation, RF importance) → four models; parity plots for each (logistic regression/kNN/MLP/RF in the deck's figure) with metrics table.
  8. **Class imbalance, activity cliffs, applicability domain, uncertainty** (ensemble variance, conformal prediction teaser).
  9. **Interpretation:** feature importance, SHAP on fingerprints mapped back to substructures (draw top bits), partial dependence for descriptors.
  10. **Reporting checklist:** data source/versions, curation, splits, metrics with CIs (bootstrap), baselines, limitations, reproducibility (seeds, environment file).
- **Exercises:** *Project warm-up* — train and evaluate a model for AChE actives vs inactives using three feature sets; submit a one-page model card.
- **Pitfalls:** **data leakage** (scaling/feature selection before splitting, duplicate molecules across folds, near-duplicate analog series), reporting accuracy on imbalanced data, tuning on the test set, comparing to no baseline.

#### nb14 · 3D-QSAR (CoMFA-style) and field descriptors  ●●● · advanced/optional · ~120 min
- **Slide source:** 06 QSAR (2D/3D descriptors; GRID and CoMFA; CoMFA/CoMSIA case on aldo-keto reductase YtbE with virtual screening).
- **Objectives:** understand field-based 3D-QSAR (alignment, grid probes, PLS) and implement a minimal version.
- **Content plan:**
  1. Concept: molecules placed in a 3D lattice of thousands of evenly spaced grid points; probe atoms (steric, electrostatic, hydrophobic, H-bond) measure interaction energy at each node; PLS relates field values to activity; contour maps interpretation (CoMFA vs CoMSIA with Gaussian similarity indices).
  2. Build a mini-pipeline: aligned conformers (common-substructure or O3A alignment) of a congeneric series (≈ 20–30 molecules; steroid benchmark-like or provided set) → grid generation (2 Å spacing) → steric (LJ, capped) and electrostatic (Coulomb with Gasteiger charges) fields → `PLSRegression` with LOO/Leave-group-out CV to choose components → q², r² → coefficient maps → show contours in `py3Dmol`.
  3. Sensitivity analysis: alignment choice, grid offset, probe type, cut-offs; why results are alignment-dependent.
  4. Use model predictions for a mini virtual screen (as in the YtbE example: predicted substrates verified experimentally—conceptual).
  5. Compare to 2D fingerprint/ML models from nb13.
- **Pitfalls:** misalignment, grid-origin dependence, singularities near atoms (field cut-offs), overinterpretation of contour maps.

#### nb15 · Drug-likeness filters and in-silico ADME/T  ●●○ · ~90 min
- **Slide source:** 03 (SwissADME, PreADMET, drug-likeness card, home use), 07 (Lipinski, ZINC property distributions, ADMET modeling), 01 (ADME/T importance).
- **Objectives:** apply rule-based filters; interpret a SwissADME-style report; understand what each ADME endpoint means; build a simple ADME classifier.
- **Content plan:**
  1. **Lipinski Rule of Five** (≤ 5 HBD, ≤ 10 HBA, MW < 500, logP ≤ 5; ≤ 1 violation tolerated per the deck's wording) plus **Ghose, Veber (rotatable bonds, TPSA), Egan, Muegge**, bioavailability score; implement all in `chemlib` with unit tests.
  2. Reproduce a **SwissADME-style report** locally for the piperonyl amide example in the deck (SMILES `O=C(c1ccc2c(c1)OCO2)Nc1ccc2c(c1)OCO2`, MW ≈ 285.25, TPSA ≈ 66 Å²): physicochemical table, lipophilicity (Crippen vs consensus note), water solubility estimates (ESOL implementation from the Delaney equation), pharmacokinetics (GI absorption/BBB via BOILED-Egg logic using TPSA vs logP regions), medicinal-chemistry alerts (PAINS, Brenk via `FilterCatalog`), synthetic accessibility (RDKit Contrib SA score), lead-likeness.
  3. **Property distributions:** compute for a ZINC-like/ChEMBL subset and replicate the six distributions (MW, HBD, HBA, cLogP, Lipinski violations, rotatable bonds); show the fraction passing each rule.
  4. **PreADMET-style toxicity endpoints (Ames, hERG, carcinogenicity):** explain via public datasets (Tox21, hERG, Ames) → small RF classifier with scaffold split; discuss reliability.
  5. **ADMET via binding affinity to transporters/metabolizing enzymes (CYPs, P-gp):** conceptual docking-based vs ML-based modeling; link to nb13.
  6. **Multi-objective view:** Pareto front over (predicted potency, logP, TPSA, SA score); QED.
  7. Web exercise: submit 5 compounds to SwissADME and PreADMET and compare with local results (optional/offline-safe).
- **Exercises:** filter the curated ChEMBL set; plot BOILED-Egg regions; identify PAINS in actives and discuss whether to exclude them.
- **Pitfalls:** treating rules as laws (exceptions: natural products, antibiotics, PROTACs), tool-to-tool logP disagreements, applying rules to charged/salt forms.

---

### Part F · Structure-based design

#### nb16 · Target discovery and structure preparation  ●●○ · ~120 min
- **Slide source:** 07 (disease characterization via differential gene expression, target discovery, comparative/subtractive genomics, network theory, infectious vs non-infectious targets, homology modeling, binding-site location, structure preparation, protonation/tautomers).
- **Objectives:** understand how targets are chosen; obtain/model a 3D structure; locate pockets; prepare receptor and ligands for docking.
- **Content plan:**
  1. **Disease state and targets:** differential gene expression (toy RNA-seq/microarray table → t-test/limma-style volcano plot with `scipy`/`statsmodels`); aberrant expression and target regulation; examples listed in the deck (cancer: DNA topoisomerase II, PDGFR, COX-2; Alzheimer's: AChE, BACE-1, tau/amyloid aggregates; infectious: transpeptidase, lumazine synthase/riboflavin synthase with non-homologous host enzyme criteria).
  2. **Subtractive genomics (toy):** pathways of host vs pathogen as sets → unique-pathway proteins → BLAST-like non-homology filter (use precomputed identity table) → essential-gene filter (DEG-style list) → candidate list. Mention the 7-genes example for *M. tuberculosis* qualitatively.
  3. **Network-based target discovery (toy):** differential network analysis with `networkx` (normal vs disease edges; hub/bottleneck ranking).
  4. **Getting a 3D structure:** PDB download by ID (Biopython); sequence → structure options: homology modeling (SWISS-MODEL), de novo (Rosetta), deep learning (AlphaFold DB download); compare confidence metrics (pLDDT) and when each is acceptable. Cryo-EM resolution context.
  5. **Binding-site detection:** probe-based approaches (small organic probes such as ethanol, acetamide, urea, phenol → overlapping spheres), geometric pocket finders (CASTp, fpocket), multiple/cryptic sites (sirtuin, tau fibril examples), open vs closed conformations (sirtuin-1/2/3).
  6. **Receptor preparation checklist:** remove waters (except structurally important), alternate locations, add hydrogens, assign protonation states of Asp/Glu/His/Lys (His tautomers HID/HIE/HIP depending on environment and pH), cap termini, fix missing atoms (PDBFixer), assign charges; save as PDB → PDBQT (Meeko/ADFR).
  7. **Ligand preparation:** protonation at pH 7.4, tautomers, stereoisomer enumeration, 3D conformers, charges (links to nb05).
  8. **Binding-site definition:** grid box from co-crystallized ligand centroid (+ padding) vs pocket finder output.
- **Data:** a small, well-characterized complex (e.g. BACE-1 or AChE with a bound inhibitor; select a PDB entry at build time and record it in `data/README.md`).
- **Exercises:** prepare a receptor from a raw PDB, compare two protonation schemes, and explain which residues change; compute pocket volume for three predicted pockets.
- **Pitfalls:** wrong His tautomer, leaving crystallographic waters that block the pocket, mixing chains of an oligomer, missing cofactors/metals.

#### nb17 · Scoring functions and conformational sampling  ●●● · ~120 min
- **Slide source:** 07 (small-molecule libraries, free energy, scoring functions, AutoDock energy, Glide score, sampling algorithms, Monte Carlo SA), 03 (scoring function forms, docking software list).
- **Objectives:** relate scoring-function terms to physics; implement a toy scoring function and sampler; understand why scores are not free energies.
- **Content plan:**
  1. **Binding free energy:** ΔG = −RT ln K_a = RT ln K_i; relationship to IC50 (Cheng–Prusoff); realistic accuracy of docking scores (qualitative).
  2. **Three classes of scoring functions:** physics-based (force-field vdW 12–6, H-bond 12–10, Coulomb with distance-dependent dielectric, desolvation), empirical (weighted sums fitted to affinity data; e.g. terms W_vdw, W_HB, W_elec, W_ds), knowledge-based (potentials of mean force, sf = Σ −k_BT ln g(r_ij)). Plot each functional form.
  3. **AutoDock energy decomposition:** E = E_vdW + E_Hbond + E_elec + E_internal (ligand internal energy change upon binding); code a mini-implementation for a ligand + small pocket of point atoms (toy), reproduce the "332.0·q_i q_j/(ε(r)·r)" Coulomb constant and discuss units (kcal/mol, Å, e).
  4. **Glide-style terms (conceptual):** H-bond term with distance/angle penalty functions g(Δr) (1 near ideal, e.g. < ~1.8 Å, decaying to 0) and h(Δα) (1 near 180°), hydrophobic enclosure, rotatable-bond penalty; implement the piecewise functions and plot.
  5. **Sampling algorithms:** systematic (combinatorial, incremental construction), stochastic (Monte Carlo/simulated annealing, genetic algorithm), deterministic (minimization, MD). Implement **Metropolis Monte Carlo** for a rigid ligand in a 2D/3D toy potential: random translation/rotation/torsion, accept if ΔE < 0 else accept with probability exp(−ΔE/k_BT); show annealing schedule effect on finding the global minimum; visualize trajectories.
  6. **Flexibility:** ligand flexibility (torsions), receptor flexibility (side-chain rotamers, ensemble docking), induced fit; discuss "native conformation of ligand differs from bound": strain energy calculation (MMFF energy of bound pose minus relaxed minimum). The deck's discussion prompt (are the structural changes stabilizing or destabilizing?) → short written reflection.
  7. **Docking software landscape:** table (DOCK, FlexX, Hammerhead, Surflex, SLIDE, AutoDock/Vina, ICM, MCDock, GOLD, GemDock, FRED, Glide, Yucca) with sampling algorithm and scoring type; they differ in scoring function, ligand/receptor flexibility and minimization algorithms.
  8. **Beyond docking scores:** MM-GBSA/MM-PBSA, free-energy perturbation (conceptual); data-driven alternatives (ML scoring) vs MD-based free-energy methods (computational cost trade-off).
- **Exercises:** implement a 12-6 Lennard-Jones + Coulomb score for two molecules; tune the MC temperature schedule; compute ΔG from K_i and back.
- **Pitfalls:** unit mix-ups, hard-sphere clashes, treating scores as absolute energies, scoring-function bias towards large ligands.

#### nb18 · Docking and virtual screening  ●●● · ~150 min
- **Slide source:** 07 (docking validation by RMSD, hands-on BACE-1, VS vs HTS example, CADD success stories), 03 (virtual screening).
- **Objectives:** run docking with AutoDock Vina; validate a protocol by redocking and by correlating with experiment; run and evaluate a small VS campaign; interpret results critically.
- **Libraries:** `vina`, `meeko`, `rdkit`, `biopython`, `pandas`, `scikit-learn`, `py3Dmol`.
- **Content plan:**
  1. **Pipeline diagram:** target prep (nb16) → ligand library prep → docking → scoring/ranking → pose inspection → post-filters (ADME, PAINS) → experimental testing.
  2. **Ligand library prep:** SMILES → protonated 3D → PDBQT with Meeko. *Slide path used `obabel → mol2 → pdbqt via MGLTools`; implement the Meeko route and optionally show the legacy path.*
  3. **Redocking validation:** dock the co-crystallized ligand; compute **heavy-atom RMSD** (symmetry-aware, e.g. `rdMolAlign.CalcRMS`) between docked and crystal poses; success criterion RMSD < 2 Å; repeat across exhaustiveness settings and random seeds; report variability.
  4. **Cross-validation with experiment:** dock ≈ 30–100 BACE-1 (or AChE) ligands with known K_i/IC50 (from ChEMBL); convert docking score to predicted K_i (ΔG = RT ln K_i) and compare with experimental values: Pearson/Spearman, RMSE in log units, rank-ordering ability. *Hands-on from the deck: download BACE-1 actives from ChEMBL → SMILES → mol2 (Open Babel) → pdbqt → dock → compare → "what do you learn?"* — prepare the reflection prompts (scoring-function limits, flexible receptors, protonation of the catalytic Asp dyad, charged ligands).
  5. **Virtual screening vs HTS:** quantitative illustration of hit rates: HTS of 400,000 molecules → 85 hits (0.021%) vs VS of 235,000 molecules → 365 tested, 127 hits (34.8%) for PTP1B (Doman et al., 2002) — plot and discuss why VS enrichment arises and the caveat of selection bias.
  6. **Enrichment analysis:** actives + decoys docked; ROC, AUC, EF1%, BEDROC, enrichment plots; compare Vina vs a similarity baseline (nb10) vs ML (nb13) — "consensus" ranking via rank-averaging.
  7. **Pose analysis:** interaction fingerprints (PLIP/ProLIF or own distance rules), key H-bond contacts, clash check, ligand strain.
  8. **Post-processing:** cluster hits by scaffold, apply ADMET filters (nb15), select diverse top-N for "purchase".
  9. **Success stories** (qualitative table): dorzolamide (carbonic anhydrase inhibitor), captopril (ACE inhibitor), saquinavir/ritonavir/indinavir (HIV protease), tirofiban (fibrinogen antagonist), zanamivir (influenza neuraminidase inhibitor).
  10. **Scaling:** `FULL_RUN` switch; multiprocessing; HPC job-array template in `docs/`.
- **Exercises:** redock three different ligands and report RMSD; change the box size and observe score/pose change; screen a 500-compound subset and compute EF1%.
- **Pitfalls:** box too small/large, ligand protonation, ignoring cofactors/water, over-interpreting small score differences (error of a few kcal/mol), rank vs absolute agreement.

#### nb19 · De novo design and fragment-based drug discovery  ●●○ · ~120 min
- **Slide source:** 03 (de novo design, fragment-based drug discovery vs HTS, rule of three, FBDD success stories, link/grow, LigBuilder 2).
- **Objectives:** explain de novo design workflows and fragment strategies; build a fragment library with the rule of three; generate candidates with fragment growing/linking; filter candidates for properties and synthesizability.
- **Content plan:**
  1. **De novo design logic:** identify the pocket → identify charged groups, H-bond donors/acceptors, hydrophobic/hydrophilic regions, π-stacking possibilities → build complementary molecules (atom-by-atom or fragment-based). Programmatic pocket-property map (donor/acceptor/hydrophobic points as a "receptor property" diagram; recreate with matplotlib).
  2. **Fragment placement → connection with skeletons → "real molecule" creation** (the two-stage schematic in the deck): implement a toy version using BRICS fragmentation of known ligands + recombination (`BRICS.BRICSBuild`) and scoring by a pharmacophore/feature-complement function.
  3. **Fragment-based drug discovery vs HTS:** conceptual simulation of library size and hit rates; fragments hit weakly (mM–µM) but with high ligand efficiency; compute **ligand efficiency (LE = −ΔG/N_heavy)** and lipophilic efficiency (LLE = pIC50 − logP).
  4. **Rule of three** (Congreve et al. 2003): MW ≈ 300 Da; HBD ≤ 3; HBA ≤ 3; cLogP ≤ 3; rotatable bonds ≤ 3; PSA ≈ < 60 Å² (the deck prints "<"; see errata). Filter a fragment subset (e.g. ~500k ZINC fragment library mentioned in deck — use a few-thousand subset) and plot property distributions.
  5. **Link/grow strategies:** sub-pocket definitions (e.g. Ile56, Phe46, Asp37 example), fragment *growing* (add substituents with BRICS/reaction SMARTS) and *linking* (join two fragments with linkers of varied length; score distance compatibility); optional re-docking via nb18 to rank.
  6. **FBDD success stories (Erlanson et al., Nat. Rev. Drug Discov. 2016):** ~30 clinical candidates; vemurafenib (BRAF V600E, approved 2011) and venetoclax (BCL-2, approved 2016); PLX3397 (phase III at the time), AT9283 (phase II). Show the lead-fragment → drug evolution as a figure generated from SMILES (verify structures at build time).
  7. **LigBuilder 2 workflow (Yuan, Pei, Lai, 2011):** design objectives: binding-affinity prediction, physicochemical property evaluation, lock-and-key matching, synthesizability prediction. If LigBuilder is installed: run a guided tutorial on a small pocket; otherwise open-source substitute notebook cells (BRICS growing + scoring + SA score) and clearly label as a simplified analog.
  8. **Candidate triage:** Lipinski/rule-of-3 appropriate filters, PAINS, SA score, retrosynthesis hints (optional AiZynthFinder mention only).
  9. **Generative models teaser (optional):** SMILES RNN/VAE or fragment-based generator; evaluation metrics (validity, uniqueness, novelty, diversity, property distribution).
- **Exercises:** *Assignment A2* — design ≥ 20 de novo candidates for a given pocket (LigBuilder or the notebook pipeline), dock them (nb18), and rank by predicted affinity, LE and synthesizability; submit a short report.
- **Pitfalls:** chemically unreasonable fragments linked, ignoring synthesizability, scoring-function exploitation, fragments with promiscuous/PAINS motifs.

---

### Part G · Capstone and advanced

#### nb20 · Project template: ligand-based model for a biomolecular target  ●●● · self-paced
- **Slide source:** 01 (project definition), 03, 05, 06 (all modeling tools).
- **Objectives:** provide a fully worked, parameterized notebook that students copy and adapt to their chosen target.
- **Content plan:**
  1. `TARGET_CHEMBL_ID`, activity type, thresholds as parameters at top.
  2. Data retrieval + curation (nb06 pipeline).
  3. EDA: activity distribution, chemical-space map (PCA/t-SNE/UMAP of ECFP), scaffold diversity, property profiles.
  4. Feature sets: ECFP4, MACCS, RDKit/Mordred descriptors, pharmacophore fingerprint.
  5. Splits: random, scaffold, (time if available); nested CV.
  6. Models: ≥ 3 algorithms + baselines; calibration; uncertainty.
  7. Evaluation and error analysis (activity cliffs, mis-predicted scaffolds).
  8. Model interpretation and applicability domain.
  9. **Prospective step:** score a held-out external library (e.g. ZINC subset), filter by ADME rules (nb15), pick top 20, optionally dock (nb18) and compare ranks.
  10. Export: model file (`joblib`), `model_card.md` (intended use, data, metrics, limitations), predictions CSV.
- **Deliverables:** notebook, model card, 4–6 page report (`project/report_template.md`), 5-minute presentation.

#### nb21 · Graph neural networks for molecular property prediction  ●●● · advanced/optional
- **Slide source:** 02 (solvation energy prediction using GNN); 05/04 (graphs, adjacency).
- **Objectives:** connect molecular-graph representations to message-passing networks; train a small GNN; compare with fingerprint + RF baselines.
- **Content plan:**
  1. Molecular featurization: node features (element, degree, aromaticity, hybridization, formal charge), edge features (bond type); dataset class (PyTorch Geometric or pure PyTorch).
  2. Message passing from scratch on the adjacency matrix (nb04 teaser) → GCN/GIN/MPNN layers; global pooling readout.
  3. Task: aqueous solvation free energy (FreeSolv) or solubility (ESOL) regression; scaffold split; training loop with early stopping; learning curves.
  4. Baselines: ECFP + RF/GP, descriptor + ridge; compare RMSE with CIs.
  5. Explainability: atom-level attributions (gradient × input) drawn on structures.
  6. Limits: small data regimes, out-of-distribution scaffolds, cost vs gain.
- **Pitfalls:** random vs scaffold splits inflate performance, over-parameterization on < 1,000 molecules, seed variance.

---

## 8. Assignments, project and assessment

### 8.1 Assignment A1 — Searching chemical libraries with a template or pharmacophore (15 pts of the 30)
- **Inputs:** target + 5–15 known actives (provided), a 20k–50k-compound library (ZINC/ChEMBL subset), starter notebook.
- **Tasks:** (1) curate actives and library; (2) substructure/SMARTS template search; (3) pharmacophore model (2D fingerprint and 3D feature-based); (4) similarity search baseline; (5) evaluate with decoys (EF1%, ROC-AUC); (6) select and justify top-10 diverse hits.
- **Rubric:** correctness of data handling 20%, search implementation 25%, pharmacophore quality and rationale 20%, evaluation rigor 20%, clarity of write-up 15%.

### 8.2 Assignment A2 — De novo design for a biological target (15 pts)
- **Inputs:** prepared receptor + pocket definition, optional LigBuilder installation.
- **Tasks:** design ≥ 20 molecules; filter by drug-likeness/SA; dock; compute LE; propose top-5 with interaction analysis.
- **Rubric:** pocket analysis 20%, design strategy 20%, filtering and docking validity 25%, ranking/analysis 20%, communication 15%.

### 8.3 Warm-up (ungraded) tasks taken directly from the slides
- Draw the 10 SMILES; write SMILES for the 6 molecules (nb03).
- Build methane by Z-matrix (nb02).
- Install Molden, Open Babel, RDKit, Checkmol; build ethyl chloride as `.mol2`; download YMDB or DrugBank and report molecule count and fields; obabel vs RDKit 3D coordinates; functional groups + FG fingerprint (nb05, nb07).
- Download human AChE actives from ChEMBL → CSV → SMILES + activity CSV → MOL and MOL2 (nb06).
- BACE-1 virtual screening hands-on (nb18).

### 8.4 Project (20 pts)
- Ligand-based ML model for inhibitors of a biomolecular target of the student's choice (approved by instructor). Built from `nb20`. Rubric in `project/rubric.md`: data quality 20%, modeling and validation 30%, interpretation 20%, reproducibility 15%, communication 15%.

### 8.5 Mid-term and end-term coverage (suggested)
- **Mid-term (20):** Parts A–C (representations, databases, functional groups, descriptors, fingerprints): short derivations, SMILES writing, matrix computations, interpreting RDKit output.
- **End-term (30):** Parts D–F (search, pharmacophore, QSAR/ML, ADME, docking/VS, de novo): metric calculations, workflow design, critique of a published VS/QSAR study.
- Provide a question bank per notebook in `docs/question_bank.md` (MCQ + numeric + short answer; at least 10 per notebook).

---

## 9. Dataset registry

Record all of these in `data/README.md` with URL, access date, licence, citation, SHA-256 and the notebook(s) using them. **Do not commit large raw files**; fetch with `data/fetch_data.py` and ship small processed samples.

| ID | Content | Source (verify at build time) | Used in | Size target |
|---|---|---|---|---|
| D01 | Slide SMILES lists (10 draw, 6 write) | this spec | nb03 | tiny |
| D02 | Alkane isomers + boiling points | literature/NIST WebBook | nb01, nb08 | < 100 rows |
| D03 | Small SDF of ~50 drugs with 3D coordinates | PubChem/DrugBank open subset | nb05, nb07 | 50 |
| D04 | ChEMBL human AChE (CHEMBL ID resolved by API) | ChEMBL | nb06, nb10, nb13 | thousands (cache) |
| D05 | ChEMBL BACE-1 | ChEMBL | nb08, nb13, nb18 | ~1–5k |
| D06 | ChEMBL MAO-B K_i | ChEMBL | nb13 | ~1k |
| D07 | P-gp substrates/inhibitors (public alternative) | ChEMBL/TDC | nb07, nb13 | ~2k |
| D08 | YMDB or DrugBank open structure file | YMDB / DrugBank | nb05, nb06 | sample only |
| D09 | ZINC subset (drug-like 20–50k; fragment subset) | ZINC | nb10, nb15, nb19, A1 | 20–50k (cache) |
| D10 | PDB entries for BACE-1 and/or AChE complexes | RCSB PDB | nb16, nb18 | few files |
| D11 | AlphaFold model of a target (for pLDDT demo) | AlphaFold DB | nb16 | 1 file |
| D12 | Toy gene-expression table + pathway sets | synthetic generator in repo | nb16 | tiny |
| D13 | Hansch-type series (logP, σ, activity) | literature or simulated | nb12 | < 100 |
| D14 | Free-Wilson R-group table | literature or simulated | nb12 | ~20 |
| D15 | Torsion data (O–C–C–O, C–C–C–C) | CSD-derived summaries or computed | nb06 | small |
| D16 | Tox/ADME sets (Tox21, hERG, Ames, FreeSolv, ESOL) | MoleculeNet/TDC | nb15, nb21 | various |
| D17 | Decoy sets (DUD-E-style or property-matched) | generated | nb10, nb18 | per target |
| D18 | Congeneric series with activities for 3D-QSAR | literature/provided | nb14 | 20–30 |

---

## 10. Tool registry (slide tool → Python equivalent)

| Tool in slides | Purpose | In this repo |
|---|---|---|
| XDrawChem, ISIS Draw, ChemDraw, JME, Marvin Sketch, PubChem sketcher | 2D sketching | Conceptual (nb02); students export SMILES/MOL; RDKit depiction in code |
| Molden, Avogadro, Maestro, VMD, PyMOL, Chimera | 3D build and visualization | `py3Dmol`/`nglview` in notebooks; GUI tools as optional exercises |
| Open Babel (`obabel`) | format conversion, 3D gen, FP2, charges | nb00, nb05 (CLI + Python) |
| RDKit | core toolkit | all notebooks |
| CORINA | 2D → 3D (commercial) | conceptual; RDKit ETKDG/Open Babel used |
| Checkmol | functional groups, FG fingerprints | nb07 (+ SMARTS fallback) |
| PaDEL-Descriptor | 797 descriptors + 10 FP types (CLI/GUI) | nb08 via `padelpy`/CLI |
| Mordred | descriptors (CLI `python -m mordred file.smi -s -o out.csv`) | nb08 |
| MayaChemTools, Dragon, ChemDes | descriptors/fingerprints/servers | conceptual (nb08) |
| SwissADME, PreADMET | ADME/tox servers | nb15 web exercise + local equivalents |
| SwissSimilarity | LBVS web tool | nb10 web exercise |
| WEKA | ML in cited studies | scikit-learn replacements (nb13) |
| CASTp, fpocket | binding-site detection | nb16 |
| SwissModel, Rosetta, AlphaFold | structure modeling | nb16 (AlphaFold DB download, others conceptual) |
| AutoDock/Vina, MGLTools | docking and PDBQT prep | nb18 via `vina` + `meeko` |
| DOCK, FlexX, GOLD, Glide, FRED, Surflex, ICM, … | docking software | landscape table (nb17) |
| LigBuilder 2 | de novo design | nb19 (guided, optional) + BRICS substitute |
| DRAGON/GRID/CoMFA/CoMSIA | field descriptors / 3D-QSAR | nb14 (mini implementation) |

---

## 11. Slide errata and modernization notes

The agent should implement the corrected versions and include a *"Note vs. slides"* cell where flagged. Items marked ⚠ should be double-checked against primary sources before publishing.

| # | Deck / topic | Issue | Handling |
|---|---|---|---|
| E1 | 03 Tools — force-field slide | Text says non-bonded (vdW, electrostatics) terms are "not included" in structure optimization; common force fields used for this (MMFF94, UFF) *do* include non-bonded terms (for atoms separated by enough bonds). | nb05: show energy decomposition from RDKit/Open Babel and state precisely what is included ⚠ |
| E2 | 03 Tools — "SMILES to Mol2 using rdkit" | Code writes a `.mol` via `SDWriter`, not MOL2; RDKit has no native MOL2 writer; `charged_mol_3D = uncharged_mol_3D` is an alias (not a copy); Gasteiger charges are atom properties not stored in MOL blocks. | nb05: use `obabel` for MOL2, `copy.deepcopy`/`Chem.Mol(m)`, write charges explicitly |
| E3 | 06 Similarity — RDKit Tanimoto snippet | `GetMorganFingerprintAsBitVect` is deprecated in recent RDKit; the example SMILES `O=C1C2CCCC2N1Cc3cccc3` contains a five-atom aromatic ring and should fail to parse/kekulize (likely intended a benzyl group, `Cc3ccccc3`). Radius 3 is ECFP6, not "ECFP4". | nb10: use `rdFingerprintGenerator`, fix SMILES, clarify radius naming ⚠ verify parse behavior |
| E4 | 06 Similarity — FP2 | Text says FP2 has 1,024 bits (Open Babel); the summary table lists 256 features. | nb09: compute actual lengths; report 1,024-bit FP2 and explain the table's different counting ⚠ |
| E5 | 06 Similarity — Tversky | Index written as c/(αa+βb+c) with *a, b = unique* counts, whereas the coefficient slide defines *a = shared*. | nb10: use distinct symbols (`n_shared`, `n_only_x`, `n_only_y`) throughout |
| E6 | 06 Similarity — contingency example | Numeric table (a=4, b=2, c=1, d=3) should be recomputed from the printed vectors; verify consistency. | nb10: compute from vectors and display |
| E7 | 06 Similarity — complexity | "NP-complete / NP-hard" used loosely for graph isomorphism. Subgraph isomorphism is NP-complete; general graph isomorphism is not known to be NP-complete (quasi-polynomial algorithm by Babai, 2015). | nb04/nb10: state precisely |
| E8 | 06 QSAR & 01/04 — F1 score | F1 defined with specificity and sensitivity; standard F1 is the harmonic mean of **precision** and **recall (sensitivity)**. | nb13: use standard definition; add a "Note vs. slides" cell |
| E9 | 05 Descriptors — LUMO | Defined as "lowest *occupied*" MO; should be lowest *unoccupied*. | nb08: correct |
| E10 | 03 Tools — rule of three | Slide uses "<3" for several limits; the original rule is "≤ 3" (MW ≈ 300). | nb19: use original definition, cite Congreve 2003 |
| E11 | 07 VS — docking MGLTools path | MGLTools (Python 2 era) is legacy for PDBQT prep. | nb18: Meeko/ADFR; optional legacy note |
| E12 | 07 VS/Glide slide | Wording of g(Δr)/h(Δα) differs between decks (1.8 Å threshold). | nb17: implement as documented in Friesner et al. 2004; cite equation ⚠ |
| E13 | 02/04 — isomer counts | Counts shown for alkane skeletal isomers (C7 = 9 … C40 = 6.2×10^13); clarify they are acyclic constitutional isomers (stereoisomers excluded). | nb01/nb07: state scope; verify with enumeration for small n |
| E14 | 01/03 — typos | "Chem. Inj Comput. Sci." (J. Chem. Inf. Comput. Sci. 1988, 28, 31–36), "Sterochemistry", "Drug-likelyness", "De Nova", "on-bonded", "kappa nearest neighbor" (= k-nearest neighbor). | Use correct spellings in all notebooks |
| E15 | 03 Tools — web links/servers | Servers and URLs (SwissADME, PreADMET, ChemDes, CASTp, SwissSimilarity, planetcalc, ChemMine tools) may have moved. | Verify every link at build time; provide offline fallbacks |
| E16 | 04 — oxidation example "benzene hydrophilicity index" etc. | Some empirical formulas appear only as images in the decks. | Re-derive/verify from primary literature before coding; do not guess |

---

## 12. References

**Textbooks (from the course introduction)**
- Leach, A. R.; Gillet, V. J. *An Introduction to Chemoinformatics*. Springer.
- Karthikeyan, M.; Vyas, R. *Practical Chemoinformatics*. Springer.
- Morrison, R. T.; Boyd, R. N. *Organic Chemistry*. Pearson India.
- Todeschini, R.; Consonni, V. *Handbook of Molecular Descriptors* / *Molecular Descriptors for Chemoinformatics*. Wiley-VCH.

**Papers cited in the slides**
- Weininger, D. SMILES, a chemical language and information system. *J. Chem. Inf. Comput. Sci.* **1988**, 28, 31–36.
- O'Boyle, N. M. et al. Open Babel: An open chemical toolbox. *J. Cheminformatics* **2011**, 3, 33.
- Yap, C. W. PaDEL-descriptor. *J. Comput. Chem.* **2011**, 32, 1466–1474.
- Moriwaki, H. et al. Mordred: a molecular descriptor calculator. *J. Cheminformatics* **2018**, 10, 4.
- Congreve, M. et al. A "rule of three" for fragment-based lead discovery? *Drug Discov. Today* **2003**, 8, 876–877.
- Erlanson, D. A. et al. Twenty years on: the impact of fragments on drug discovery. *Nat. Rev. Drug Discov.* **2016**, 15, 605–619.
- Jhoti, H. et al. / Hajduk & Greer. Link/grow strategies in FBDD. *Nat. Rev. Drug Discov.* **2005**, 4, 649.
- Yuan, Y.; Pei, J.; Lai, L. LigBuilder 2: a practical de novo drug design approach. *J. Chem. Inf. Model.* **2011**, 51, 1083–1091.
- Friesner, R. A. et al. Glide: a new approach for rapid, accurate docking and scoring. *J. Med. Chem.* **2004**, 47, 1739–1749.
- Doman, T. N. et al. Molecular docking and high-throughput screening for novel inhibitors of protein tyrosine phosphatase-1B. *J. Med. Chem.* **2002**, 45, 2213–2221.
- Bragina, M. E. et al. The SwissSimilarity 2021 web tool. *Int. J. Mol. Sci.* **2022**, 23, 811.
- Daina, A.; Michielin, O.; Zoete, V. SwissADME. *Sci. Rep.* **2017**, 7, 42717.
- Murugan, N. A. et al. *Pharmaceuticals* **2022**, 15(1), 63 (scoring functions and small-molecule databases); *Int. J. Mol. Sci.* **2020**, 21(20), 7648 (ML for MAO-B affinity); *ACS Chem. Neurosci.* **2018**, 9, 1757 (multiple binding sites); *Eur. J. Nucl. Med. Mol. Imaging* **2019**, 46, 1369 (docking validation).
- Ullmann, J. R. An algorithm for subgraph isomorphism. *J. ACM* **1976**; Cordella, L. P.; Foggia, P. et al. (VF/VF2) *IEEE TPAMI* **2004**.
- Hansch, C.; Fujita, T. *J. Am. Chem. Soc.* **1964**; Free, S. M.; Wilson, J. W. *J. Med. Chem.* **1964**.
- Chen, X.; Reynolds, C. H. Performance of similarity measures in 2D fragment-based similarity searching. *J. Chem. Inf. Comput. Sci.* **2002**; Todeschini, R. et al. Similarity coefficients for binary chemoinformatics data. *J. Chem. Inf. Model.* **2012**, 52, 2884 (the 51-coefficient survey; ⚠ verify).

*(Verify all bibliographic details against the publishers' pages at build time; entries above are reconstructed from the slides.)*

---

## 13. Definition of done and agent prompt template

### 13.1 Definition of done (per notebook)

- [ ] Follows the cell skeleton in Section 5.1; title, objectives, prerequisites, summary, references present.
- [ ] Runs top-to-bottom in a clean environment in < 5 min with `FULL_RUN=False`; no hidden state.
- [ ] No outputs committed (`nbstripout`); figures regenerated by code.
- [ ] ≥ 3 exercises with `assert` self-checks; matching solution notebook in `solutions/`.
- [ ] Every function used in ≥ 2 notebooks lives in `src/chemlib/` with a unit test.
- [ ] All errata items relevant to the notebook are handled and annotated.
- [ ] All external data are cached with provenance in `data/README.md`; offline fallback exists.
- [ ] Spell-check and `ruff` pass; no dead links.
- [ ] CI green; entry added to `docs/CHANGELOG.md` and `docs/slides_index.md`.

### 13.2 Prompt template for an agent building one notebook

```text
You are building notebook <NBxx> for the repository described in cheminformatics.md.

1. Read Sections 1, 4, 5, 9, 10 and 11 of cheminformatics.md, and the spec block for <NBxx> in Section 7.
2. Create notebooks/<part>/<NBxx>_<slug>.ipynb implementing every item in the "Content plan" in order.
   - Follow the standard cell skeleton and style rules.
   - Move reusable code to src/chemlib/ and add tests under tests/.
   - Apply all relevant errata from Section 11 (add "Note vs. slides" cells).
   - Use only datasets listed in Section 9; if one must be created, add it to the registry.
3. Create solutions/<part>/<NBxx>_<slug>.ipynb with worked solutions for all exercises.
4. Execute the notebook headless (nbclient) with FULL_RUN=False; fix errors and runtime > 5 min.
5. Update docs/slides_index.md, docs/CHANGELOG.md, data/README.md.
6. Report: files created, runtime, assumptions made, any spec item you could not satisfy (and why),
   and any external resource/URL that was unreachable or had changed.

Constraints: CPU-only; deterministic seeds; no reproduction of slide images or long verbatim slide text;
all chemistry claims must be verifiable with code or cited sources.
```

### 13.3 Suggested build order and milestones

| Milestone | Notebooks | Gate |
|---|---|---|
| M1 Foundations | nb00–nb05, `chemlib.io/viz/graphs` | Env + CI green; round-trip format tests pass |
| M2 Data and groups | nb06–nb07, `chemlib.standardize` | ChEMBL AChE pipeline reproducible offline from cache |
| M3 Descriptors and FPs | nb08–nb09, `chemlib.descriptors/fingerprints` | Descriptor/FP tests; PaDEL/Mordred optional paths guarded |
| M4 Search and pharmacophore | nb10–nb11, `chemlib.pharm` | A1 starter + rubric ready |
| M5 QSAR and ADME | nb12–nb15, `chemlib.metrics` | Project template (nb20) can import nb13 pipeline |
| M6 Structure-based | nb16–nb19, `chemlib.docking` | Redocking RMSD < 2 Å on the reference complex; A2 starter ready |
| M7 Capstone | nb20–nb21, project brief/rubric | Full dry-run by a reviewer |

---

*End of master specification. Keep this file updated whenever a notebook's scope changes.*