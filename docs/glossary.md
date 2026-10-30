# Cheminformatics Glossary

Comprehensive reference of core terms, mathematical definitions, and acronyms used throughout the course.

### A
- **Activity Cliff:** A pair of structurally similar molecules (high Tanimoto similarity) that exhibit a large difference in bioactivity.
- **Applicability Domain (AD):** The response surface and chemical space in which a QSAR model makes predictions with defined reliability (measured via leverage / Williams plots).
- **AutoDock Vina:** An open-source molecular docking engine utilizing an empirical scoring function with iterated local search (Monte Carlo simulated annealing).

### B
- **BEDROC (Boltzmann-Enhanced Discrimination of ROC):** A virtual screening evaluation metric that applies exponential weighting to early recognition (the top of the ranked list).
- **Bemis-Murcko Scaffold:** The core ring systems and linkers connecting them within a molecule, abstracting away side chains.
- **BRICS (Bioactive Ring/Chain System):** A retrosynthetic fragmentation approach that cleaves molecules at 16 chemically synthesizable bond environments.

### C
- **Chemical Space:** The conceptual space spanned by all possible chemically stable molecules (estimated at $\sim 10^{60}$ drug-like compounds).
- **CoMFA (Comparative Molecular Field Analysis):** A 3D-QSAR technique where aligned molecules are placed in a 3D grid, probed with steric and electrostatic probes, and correlated with activity via Partial Least Squares (PLS).
- **Crippen LogP:** An atom-based empirical prediction of octanol-water partition coefficient based on atom typing and additive contributions.

### D
- **Dice Similarity:** A binary similarity coefficient defined as $\frac{2a}{2a + b + c}$.
- **Docking Score:** An estimated binding free energy ($\text{kcal/mol}$) reflecting steric complementarity, hydrogen bonding, and torsional strain.

### E
- **ECFP (Extended Connectivity Fingerprints):** Circular topological fingerprints based on the Morgan algorithm. ECFP4 corresponds to diameter 4 (radius 2).
- **Enrichment Factor (EF):** The ratio of actives found in the top $x\%$ of a ranked virtual screening list compared to random selection.
- **ETKDG:** Experimental-Torsion Knowledge Distance Geometry; RDKit's algorithm for generating realistic 3D molecular conformers.

### F
- **FBDD (Fragment-Based Drug Discovery):** Screening very small molecular fragments ($\text{MW} \le 300$) with high ligand efficiency, followed by fragment growing or linking.
- **Free-Wilson Analysis:** A mathematical QSAR approach assuming that biological activity is the sum of independent, additive substituent contributions on a common parent scaffold.

### H
- **Hansch Analysis:** A classical QSAR equation correlating biological activity with physicochemical properties (lipophilicity $\log P$, electronic $\sigma$, and steric $E_s$ parameters).
- **Hückel's Rule:** A cyclic, planar, fully conjugated system is aromatic if it possesses $4n + 2$ $\pi$-electrons ($n \in \{0, 1, 2, \dots\}$).

### I
- **InChI (International Chemical Identifier):** A non-proprietary, layered text identifier encoding molecular connectivity, charge, stereochemistry, and isotopic composition.

### L
- **LBDD (Ligand-Based Drug Design):** Drug design strategies utilizing bioactivity and structural features of known active ligands (QSAR, similarity search, pharmacophores) without requiring the 3D target structure.
- **Ligand Efficiency (LE):** The binding free energy divided by the number of non-hydrogen (heavy) atoms: $\text{LE} = -\frac{\Delta G}{N_{\text{heavy}}}$.
- **Lipinski's Rule of Five:** Empirical guidelines for oral bioavailability: $\text{MW} \le 500$, $\text{MolLogP} \le 5$, $\text{HBD} \le 5$, $\text{HBA} \le 10$.

### M
- **MACCS Keys:** A 166-bit structural key fingerprint where each bit denotes the presence or absence of a predefined SMARTS pattern.
- **Matthews Correlation Coefficient (MCC):** A balanced classification metric robust against extreme class imbalance:
  $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
- **Meeko:** A modern Python library developed by the Forli Lab for preparing flexible receptor and ligand PDBQT files from RDKit Mol objects.
- **Molecular Walk Count (MWC):** The total number of walks of length $k$ in a molecular graph, computed as the sum of all elements in the $k$-th power of the adjacency matrix ($A^k$).

### P
- **Pharmacophore:** An ensemble of steric and electronic features (H-bond donors, acceptors, hydrophobic regions, charges) necessary to ensure optimal macromolecular interactions.
- **pIC50:** The negative base-10 logarithm of the half-maximal inhibitory concentration in molar units: $\text{pIC}_{50} = -\log_{10}(\text{IC}_{50} \text{ [M]}) = 9 - \log_{10}(\text{IC}_{50} \text{ [nM]})$.

### R
- **Randić Index ($\chi$):** A topological branching index defined as $\sum_{(i,j)} \frac{1}{\sqrt{\delta_i \delta_j}}$, correlating strongly with physical properties like boiling point.
- **RMSD (Root Mean Square Deviation):** A measure of spatial distance between atomic coordinates of two superimposed conformations.

### S
- **SBDD (Structure-Based Drug Design):** Drug design strategies guided by the 3D atomic structure of the macromolecular target (molecular docking, binding pocket modeling).
- **Self-Returning Walk Count (SRW):** The number of walks of length $k$ that start and end at the same atom, calculated as the trace of the adjacency matrix power: $\text{Tr}(A^k)$.
- **SMARTS:** SMILES Arbitrary Target Specification, a query language used to specify substructural patterns.
- **SMILES:** Simplified Molecular Input Line Entry System, a linear representation of molecular connectivity and stereochemistry.

### T
- **Tanimoto Coefficient:** The most widely used similarity coefficient for binary fingerprints: $\frac{a}{a + b + c}$.
- **Tversky Index:** An asymmetric similarity coefficient with parameters $\alpha$ and $\beta$, useful for substructure-like matching when $\alpha \ne \beta$.

### V
- **Veber's Rules:** Bioavailability criteria based on molecular flexibility: Rotatable bonds $\le 10$ and Topological Polar Surface Area ($\text{TPSA}$) $\le 140\ \text{Å}^2$.

### W
- **Wiener Index ($W$):** The first topological index, defined as the half-sum of all shortest path distances in the molecular graph: $W = \frac{1}{2}\sum_{i,j} D_{ij}$.

### Z
- **Z-Matrix:** An internal coordinate representation of 3D molecular geometry defined by bond lengths, bond angles, and dihedral angles.
- **Zagreb Indices ($M_1, M_2$):** Topological indices based on vertex degrees: $M_1 = \sum \delta_i^2$ and $M_2 = \sum_{(i,j)} \delta_i \delta_j$.
