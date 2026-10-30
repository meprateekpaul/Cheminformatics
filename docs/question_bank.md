# Cheminformatics Course Question Bank

A comprehensive bank of multiple-choice, numerical, and short-answer exam questions aligned with course lectures and laboratory notebooks.

---

## Part A: Representations, Graphs and 3D Coordinates (nb00 – nb05)

### Q1.1 (Conceptual)
**Question:** Explain why internal coordinates (Z-matrix) are invariant to rigid-body translation and rotation, whereas Cartesian coordinates are not.  
**Answer:** Cartesian coordinates $(x, y, z)$ depend explicitly on the chosen origin and coordinate axes. In contrast, a Z-matrix defines molecular geometry using relative internal degrees of freedom: interatomic bond lengths $r_{ij}$, planar bond angles $\theta_{ijk}$, and torsional/dihedral angles $\phi_{ijkl}$, all of which are intrinsic geometric properties independent of spatial placement.

### Q1.2 (SMILES Grammar)
**Question:** Which of the following SMILES strings represents an invalid chemical structure?  
A) `CCO`  
B) `c1ccccc1`  
C) `C1CC1`  
D) `c1cccc1`  
**Answer:** D. Cyclopentadiene with all aromatic lowercase atoms `c1cccc1` lacks a valid aromatic $\pi$-electron system (has 4 $\pi$-electrons, violating Hückel's $4n+2$ rule unless an explicit charge/heteroatom is present, e.g. `[cH-]1cccc1` or `c1ccoc1`).

### Q1.3 (Graph Theory & Matrices)
**Question:** For a molecule whose adjacency matrix is $A$, what do the matrix power sums $\sum_{i,j} (A^0)_{ij}$ and $\sum_{i,j} (A^1)_{ij}$ represent?  
A) Total atoms and total molecular weight  
B) Number of heavy atoms and twice the number of bonds  
C) Number of rings and number of rotatable bonds  
D) Wiener index and Zagreb index  
**Answer:** B. $A^0 = I_N$, whose sum is the trace $N$ (number of atoms). $A^1$ is the adjacency matrix where each edge contributes two 1s (one in $(i,j)$ and one in $(j,i)$), so the sum of all elements equals $2 \times |E|$ (twice the bond count).

### Q1.4 (Numerical — Wiener Index)
**Question:** Compute the Wiener index $W = \frac{1}{2}\sum_{i,j} D_{ij}$ for propane ($\text{CH}_3\text{–CH}_2\text{–CH}_3$).  
**Answer:** Propane has 3 heavy carbon atoms ($C_1, C_2, C_3$). The topological distances are: $d(1,2)=1$, $d(2,3)=1$, and $d(1,3)=2$. The sum of all unique pair distances is $1 + 1 + 2 = 4$. Therefore, $W = 4$.

---

## Part B & C: Databases, Descriptors and Fingerprints (nb06 – nb09)

### Q2.1 (Fingerprint Mathematics)
**Question:** Two 2048-bit fingerprints $x$ and $y$ have 30 bits set in common ($a=30$). Compound $x$ has 20 unique bits ($b=20$) and compound $y$ has 50 unique bits ($c=50$). Compute their Tanimoto and Dice similarities.  
**Answer:**
- $\text{Tanimoto} = \frac{a}{a + b + c} = \frac{30}{30 + 20 + 50} = \frac{30}{100} = 0.30$.
- $\text{Dice} = \frac{2a}{2a + b + c} = \frac{60}{60 + 20 + 50} = \frac{60}{130} \approx 0.4615$.

### Q2.2 (Lipinski Rule of Five)
**Question:** A lead optimization candidate has: $\text{MW} = 512\ \text{g/mol}$, $\text{cLogP} = 5.2$, $\text{HBD} = 2$, $\text{HBA} = 7$, and $\text{RotBonds} = 6$. Does this compound pass Lipinski's Rule of Five?  
**Answer:** Yes. The compound violates 2 rules ($\text{MW} > 500$ and $\text{cLogP} > 5.0$). Under the standard definition, $\le 1$ violation is tolerated, so with 2 violations it strictly violates the Rule of 5 threshold.

### Q2.3 (Descriptor Classification)
**Question:** Classify the following descriptors as 0D, 1D, 2D, or 3D:  
1. Molecular Weight (MW)  
2. Number of primary amine groups  
3. Zagreb $M_1$ index  
4. Radius of gyration ($R_g$)  
**Answer:**
1. 0D (constitution / formula based)
2. 1D (substructure / fragment count)
3. 2D (topological / graph-based)
4. 3D (conformation / coordinate-dependent)

---

## Part D & E: Similarity, Pharmacophores, and QSAR (nb10 – nb15)

### Q3.1 (Hansch Parabolic Model)
**Question:** In a Hansch QSAR analysis, biological activity follows $\log(1/C) = -0.4(\log P)^2 + 2.0\log P + 3.1$. What is the optimal lipophilicity ($\log P^*$) for maximum biological activity?  
**Answer:** Taking the first derivative with respect to $\log P$ and setting to zero:
$$\frac{d}{d\log P} \log(1/C) = -0.8\log P + 2.0 = 0 \implies \log P^* = \frac{2.0}{0.8} = 2.50.$$

### Q3.2 (Classification Metrics & Imbalance)
**Question:** A screening test on 1,000 compounds (50 actives, 950 inactives) yields $\text{TP}=40$, $\text{FN}=10$, $\text{FP}=40$, and $\text{TN}=910$. Compute Sensitivity, Precision, and Accuracy. Why is accuracy misleading here?  
**Answer:**
- $\text{Sensitivity} = \frac{40}{40 + 10} = \frac{40}{50} = 80.0\%$.
- $\text{Precision} = \frac{40}{40 + 40} = \frac{40}{80} = 50.0\%$.
- $\text{Accuracy} = \frac{40 + 910}{1000} = \frac{950}{1000} = 95.0\%$.  
Accuracy is misleading because a trivial classifier predicting all compounds as inactive would achieve $95.0\%$ accuracy while retrieving zero active drug leads.

### Q3.3 (Data Leakage in QSAR)
**Question:** Why does standard random $k$-fold cross-validation often yield overly optimistic performance estimates for chemical bioactivity datasets, and what splitting strategy addresses this?  
**Answer:** Random splitting allows closely related structural analogs (congeneric series sharing the exact same core scaffold) to appear in both training and test sets. The model simply memorizes the shared scaffold rather than learning true generalizable SAR. **Bemis-Murcko scaffold splitting** groups molecules by core scaffold and places distinct scaffolds in train vs. test, ensuring true prospective generalization.

---

## Part F & G: Molecular Docking, De Novo Design & GNNs (nb16 – nb21)

### Q4.1 (Thermodynamic Conversion)
**Question:** If an enzyme inhibitor has $K_i = 10\ \text{nM}$ at $T = 298.15\ \text{K}$, calculate the standard binding free energy $\Delta G_{\text{bind}}$ in $\text{kcal/mol}$. ($R = 1.9872 \times 10^{-3}\ \text{kcal/(mol}\cdot\text{K)}$).  
**Answer:**
$$\Delta G_{\text{bind}} = RT \ln K_i = (1.9872 \times 10^{-3}) \times 298.15 \times \ln(10 \times 10^{-9}) = 0.5925 \times (-18.4207) \approx -10.91\ \text{kcal/mol}.$$

### Q4.2 (Docking Validation)
**Question:** What constitutes a successful redocking experiment, and why is heavy-atom RMSD evaluated rather than all-atom RMSD?  
**Answer:** Redocking is considered successful when the docking algorithm reproduces the crystallographic ligand pose within a heavy-atom $\text{RMSD} \le 2.0\ \text{Å}$. Heavy atoms (C, N, O, S, halogens) are used because crystallographic X-ray structures typically do not resolve hydrogen atom positions directly (due to low electron density), and hydrogen positions depend heavily on force field protonation assumptions.

### Q4.3 (Ligand Efficiency)
**Question:** Molecule A ($\text{MW} = 250$, 18 heavy atoms) has $\Delta G = -9.0\ \text{kcal/mol}$. Molecule B ($\text{MW} = 550$, 40 heavy atoms) has $\Delta G = -12.0\ \text{kcal/mol}$. Which molecule has superior Ligand Efficiency (LE)?  
**Answer:**
- $\text{LE}(A) = -\frac{-9.0}{18} = 0.50\ \text{kcal/mol per heavy atom}$.
- $\text{LE}(B) = -\frac{-12.0}{40} = 0.30\ \text{kcal/mol per heavy atom}$.  
Molecule A has superior Ligand Efficiency ($0.50 > 0.30$), making it a more efficient chemical starting point for lead optimization.
