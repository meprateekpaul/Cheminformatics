# Slide Index and Modernization Notes

This document provides a mapping between the course lecture slide decks, the corresponding Jupyter notebooks, and modernization corrections addressing ambiguities in the original slides.

## Lecture to Notebook Mapping

| Lecture Deck | Primary Topics | Core Notebooks |
|---|---|---|
| **01 Course Introduction** | Definitions, Chemical Universe, SBDD vs LBDD, Z-matrix, SMILES basics, Drug Discovery as Optimization | `nb00`, `nb01`, `nb02`, `nb03` |
| **02 Structure Representation** | Databases, Graph Theory, Adjacency/Distance Matrices, InChI, 3D Formats, Stereochemistry | `nb03`, `nb04`, `nb05`, `nb06` |
| **03 Cheminformatics Tools** | Editors, Visualizers, Open Babel, RDKit, Descriptors, Virtual Screening, Docking, De Novo, FBDD | `nb05`, `nb08`, `nb15`, `nb18`, `nb19` |
| **04 Organic Structures & Properties** | Constitutional & Stereoisomers, Functional Groups, Hückel Aromaticity, Checkmol, FG Fingerprints, Pharmacophores | `nb07`, `nb09`, `nb11` |
| **05 Molecular Descriptors** | 0D–4D Descriptors, Topological Indices (Wiener, Zagreb, Randić), Quantum Chemical, PaDEL, Mordred | `nb08` |
| **06 QSAR & Similarity Search** | Hansch, Free-Wilson, Substructure Search, 51 Similarity Coefficients, Tanimoto, Tversky, Dice, 3D-QSAR CoMFA | `nb09`, `nb10`, `nb12`, `nb13`, `nb14` |
| **07 Virtual Screening & SBDD** | Target Discovery, Binding Site Cavities, Lipinski Rules, Scoring Functions, AutoDock Vina, Redocking Validation | `nb15`, `nb16`, `nb17`, `nb18` |

---

## Slide Errata and Modernization Guide

| # | Deck / Topic | Original Issue in Slides | Modern / Correct Implementation in Repo |
|---|---|---|---|
| **E1** | 03 Tools — Force Fields | Text suggests non-bonded terms (vdW, electrostatics) are omitted during structure optimization. | Force fields such as MMFF94 and UFF explicitly include non-bonded Lennard-Jones and electrostatic terms for 1–4 and non-bonded atom pairs. Covered in `nb05`. |
| **E2** | 03 Tools — SMILES to Mol2 | Code writes `.mol` file with `SDWriter` but titles it "to Mol2". RDKit does not have a native Tripos MOL2 writer. | Modern conversion uses Open Babel (`obabel`) CLI or Python bindings to write true MOL2 files with partial charges. Covered in `nb05`. |
| **E3** | 06 Similarity — RDKit Tanimoto | `GetMorganFingerprintAsBitVect` is deprecated; example SMILES `O=C1C2CCCC2N1Cc3cccc3` has an invalid 5-atom aromatic ring; radius 3 is called ECFP4 instead of ECFP6. | Updated to `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`. Corrected SMILES to benzyl derivative `Cc1ccccc1`. Clarified radius 2 = ECFP4, radius 3 = ECFP6. Covered in `nb09` and `nb10`. |
| **E4** | 06 QSAR — Free-Wilson Formula | Typo in formula notation: $F_1$ printed where $f_i$ or $a_i$ was intended. | Standard Free-Wilson equation implemented: $\log(1/C) = \mu + \sum a_{ij} X_{ij}$. Covered in `nb12`. |
| **E5** | 04 Structures — Rule of 3 | Typo in Polar Surface Area condition: prints "$\approx < 60$" without clear inequality. | Stated cleanly as $\text{TPSA} \le 60\ \text{Å}^2$ following Congreve et al. (2003). Covered in `nb19`. |
| **E6** | 06 Similarity — Graph Complexity | Slide states graph isomorphism is NP-complete. | Clarified that **subgraph isomorphism is NP-complete**, whereas general graph isomorphism belongs to NP-intermediate (Babai's quasi-polynomial time theorem). Covered in `nb04`. |
