# Practical Cheminformatics: Fundamentals to Computer-Aided Drug Design

[![CI](https://github.com/meprateekpaul/Cheminformatics/actions/workflows/ci.yml/badge.svg)](https://github.com/meprateekpaul/Cheminformatics/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/badge/conda-cheminf-green.svg)](environment.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A rigorous, modern, lecture-aligned cheminformatics curriculum and hands-on laboratory course covering chemical space, molecular representations, graph theory, descriptors, fingerprints, QSAR, machine learning, pharmacophores, molecular docking, and de novo drug design.

---

## 🏛️ Course Structure

The course is organized into seven sequential parts spanning 22 self-contained Jupyter notebooks with full companion solutions, assignments, and a capstone project:

```
Cheminformatics/
├── notebooks/                # Student-facing exercises with assert-based checks
│   ├── nb00_setup_and_refresher.ipynb
│   ├── nb01_chemical_space_and_drug_discovery.ipynb
│   ├── nb02_building_3d_structures.ipynb
│   ├── nb03_smiles_and_linear_notations.ipynb
│   ├── nb04_molecular_graphs_and_matrices.ipynb
│   ├── nb05_file_formats_and_conversion.ipynb
│   ├── nb06_chemical_databases_chembl_pubchem.ipynb
│   ├── nb07_organic_structures_and_functional_groups.ipynb
│   ├── nb08_molecular_descriptors.ipynb
│   ├── nb09_fingerprints_keys_paths_circles.ipynb
│   ├── nb10_substructure_and_similarity_search.ipynb
│   ├── nb11_pharmacophore_modeling.ipynb
│   ├── nb12_qsar_foundations_hansch.ipynb
│   ├── nb13_ml_qsar_classification_regression.ipynb
│   ├── nb14_3d_qsar_comfa_fields.ipynb
│   ├── nb15_druglikeness_and_admet.ipynb
│   ├── nb16_target_discovery_and_prep.ipynb
│   ├── nb17_scoring_functions_and_sampling.ipynb
│   ├── nb18_docking_and_virtual_screening.ipynb
│   ├── nb19_denovo_design_and_fbdd.ipynb
│   ├── nb20_project_template_ligand_model.ipynb
│   └── nb21_gnn_property_prediction.ipynb
├── solutions/                # Instructor reference implementations
├── src/chemlib/              # Reusable Python package (io, standardize, viz, graphs, descriptors, etc.)
├── assignments/              # Graded assignments (A1: Library Search, A2: De Novo Design)
├── project/                  # Capstone project guidelines, rubric, and report template
├── data/                     # Provenance registry and reproducible fetch scripts (D01 - D18)
└── docs/                     # Course glossary, cheat sheets, question bank, and slide errata
```

---

## 🚀 Getting Started

### 1. Environment Setup

Create and activate the dedicated `cheminf` conda environment:

```bash
# Clone the repository
git clone https://github.com/meprateekpaul/Cheminformatics.git
cd Cheminformatics

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate cheminf

# Install chemlib in editable mode
pip install -e .
```

### 2. Verify Installation

Run the unit test suite:

```bash
pytest tests/ -v
```

### 3. Fetch Data

Generate and download course datasets:

```bash
python data/fetch_data.py
```

### 4. Launch JupyterLab

```bash
jupyter lab
```

---

## 🧪 Chemlib Package Tour

`chemlib` provides lightweight, tested utilities for everyday cheminformatics:

```python
from rdkit import Chem
from chemlib import standardize, graphs, descriptors, fingerprints, metrics

# Standardize SMILES (strip salts, uncharge, canonicalize)
smi = standardize.standardize_smiles("CC(=O)[O-].[Na+]")

# Compute topological indices & matrices
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
adj_matrix = graphs.get_adjacency_matrix(mol)
w_index = graphs.wiener_index(mol)

# Calculate Lipinski & Veber rules
props = descriptors.compute_lipinski_veber(mol)

# Generate modern fingerprints and similarity coefficients
fp1 = fingerprints.get_fingerprint(mol, "ecfp4")
fp2 = fingerprints.get_fingerprint(Chem.MolFromSmiles("c1ccccc1O"), "ecfp4")
coeffs = fingerprints.compute_similarity_coefficients(fp1, fp2)
print("Tanimoto:", coeffs["Tanimoto"])
```

---

## 📚 Curriculum Overview

| Part | Notebooks | Core Topics |
|---|---|---|
| **Part A: Foundations** | `nb00` – `nb05` | Python/RDKit refresher, Chemical Universe, 3D/Z-matrix, SMILES/InChI, Molecular Graphs, 2D⇄3D Conversions |
| **Part B: Data & Chemistry** | `nb06` – `nb07` | ChEMBL/PubChem APIs, Curation Pipelines, Functional Groups (SMARTS), Checkmol, Aromaticity |
| **Part C: Representations** | `nb08` – `nb09` | 0D–4D Descriptors, Mordred, PaDEL, MACCS, Path, Circular (ECFP), Pharmacophore Signatures |
| **Part D: Searching** | `nb10` – `nb11` | Substructure Search, 51 Similarity Coefficients, LBVS, 3D Pharmacophore Modeling, Feature Alignment |
| **Part E: QSAR & ML** | `nb12` – `nb15` | Hansch Analysis, Free-Wilson, ML QSAR (RF, SVM, kNN), 3D-QSAR CoMFA, Drug-Likeness & SwissADME |
| **Part F: Structure-Based** | `nb16` – `nb19` | Target Discovery, Binding Site Cavities, Scoring Functions, AutoDock Vina, Redocking, De Novo / FBDD |
| **Part G: Capstone** | `nb20` – `nb21` | End-to-End Ligand Modeling Pipeline, Graph Neural Networks (GNN) on Molecular Graphs |

---

## ⚖️ License & Attribution

This course material is distributed under the MIT License. Created by Prateek Paul.
