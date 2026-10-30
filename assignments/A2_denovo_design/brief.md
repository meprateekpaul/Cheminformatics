# Assignment A2: Structure-Based De Novo Design and Fragment-Based Discovery

**Points:** 15 Points (out of 30 for course assignments)  
**Deliverables:** Completed Jupyter notebook with code, candidate structures, docked poses, and written evaluation.

---

## 🎯 Objectives
1. Analyze a target binding pocket and characterize complementary chemical feature points (H-bonds, hydrophobic sub-pockets).
2. Apply the **Rule of Three** to assemble and filter a fragment starting library.
3. Generate novel de novo lead candidates using fragment growing and linking via BRICS rules.
4. Dock the designed candidates into the target pocket using AutoDock Vina.
5. Compute **Ligand Efficiency (LE)** and **Lipophilic Efficiency (LLE)** to prioritize the top 5 candidates.

---

## 📋 Tasks & Workflow
1. **Binding Pocket Characterization:** Inspect the target pocket (AChE 1EVE or BACE-1 1FKN). Map critical catalytic residues and key interaction points.
2. **Fragment Library Assembly:** Filter a fragment library using Congreve's Rule of Three: $\text{MW} \le 300$, $\text{cLogP} \le 3$, $\text{HBD} \le 3$, $\text{HBA} \le 3$, $\text{RotBonds} \le 3$, $\text{TPSA} \le 60\ \text{Å}^2$.
3. **Fragment Growing / Linking:** Using RDKit's BRICS decomposition and reaction SMARTS, grow or link starting fragments to construct $\ge 20$ complete de novo candidates.
4. **Medicinal Chemistry Filtering:** Eliminate molecules with PAINS motifs, reactive warheads, or poor Synthetic Accessibility (SA Score $> 4.5$).
5. **AutoDock Vina Docking:** Convert candidates to 3D conformers with ETKDG + MMFF, prepare PDBQTs via Meeko, and dock into the receptor pocket.
6. **Efficiency Metrics & Pose Analysis:** Compute Ligand Efficiency ($\text{LE} = -\Delta G / N_{\text{heavy}}$) and Lipophilic Efficiency ($\text{LLE} = \text{pIC}_{50} - \text{cLogP}$). Highlight key protein-ligand hydrogen bonds and hydrophobic contacts.

---

## ⚖️ Rubric
- **Pocket analysis & feature mapping:** 20%
- **De novo design strategy & BRICS workflow:** 20%
- **Filtering, drug-likeness & SA scoring:** 25%
- **Docking validity & efficiency ranking (LE/LLE):** 20%
- **Pose interpretation & communication:** 15%
