import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb19():
    nb = create_notebook()

    add_md(nb, """# Notebook 19:\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb19_denovo_design_and_fbdd.ipynb)\n\n De Novo Design and Fragment-Based Drug Discovery

**Part F: Structure-Based Drug Design (SBDD)**  
**Estimated Time:** ~120 min  
**Level:** Intermediate (●●○)

---

## 🎯 Learning Objectives
1. **De Novo Drug Design Principles:** Understand how chemical structures are built *de novo* from scratch to complement receptor pocket topology and electrostatic fields.
2. **Fragment-Based Drug Discovery (FBDD) vs HTS:** Contrast low-molecular-weight fragment screening with traditional High-Throughput Screening (HTS) in sampling chemical space.
3. **Congreve's Rule of Three:** Apply the **Rule of Three (Ro3)** for fragment library curation ($\text{MW} \le 300$, $c\log P \le 3$, $\text{HBD} \le 3$, $\text{HBA} \le 3$, $\text{RotB} \le 3$, $\text{TPSA} \le 60\ \text{Å}^2$).
4. **Ligand Efficiency Metrics:** Evaluate lead candidates using **Ligand Efficiency (LE)** and **Lipophilic Efficiency (LLE / LipE)**.
5. **Clinical Case Studies:** Trace the fragment-to-drug evolution of FDA-approved drugs: **Vemurafenib** (BRAF V600E) and **Venetoclax** (BCL-2).
6. **BRICS Fragment Assembly & Growing:** Decompose molecules into retrosynthetically sensible fragments and execute fragment *growing* and *linking* using RDKit BRICS.
7. **Medicinal Chemistry Triage:** Screen de novo designs against Pan-Assay Interference Compounds (PAINS) and estimate Synthetic Accessibility (SA Score).
""")

    add_md(nb, """## 1. Prerequisites & Environment Setup""")

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

# Path adjustments
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, BRICS
    import chemlib
    from chemlib import descriptors
    import rdkit
    print(f"RDKit {rdkit.__version__} and chemlib successfully loaded.")
except ImportError:
    print("Running in standard environment.")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 2. Fragment-Based Drug Discovery (FBDD) vs. High-Throughput Screening (HTS)

The theoretical space of drug-like molecules ($\text{MW} \le 500$) is estimated at **$10^{60}$ compounds**—astronomically larger than any physical compound collection ($10^6$ in typical HTS libraries).

### Why FBDD Works:
- **Higher Hit Rates:** Fragment libraries contain small, low-complexity molecules ($\text{MW} \sim 150 - 250$). Simple molecules have fewer opportunities for steric clashes, leading to higher hit rates ($1 - 10\%$ in FBDD vs $0.01 - 0.1\%$ in HTS).
- **High Atom Economy & Ligand Efficiency:** Each heavy atom in a validated fragment contributes substantially to binding affinity.
- **Congreve's Rule of Three (Ro3) (Congreve et al., 2003):**
  $$\text{MW} \le 300\ \text{Da}, \quad c\log P \le 3.0, \quad \text{HBD} \le 3, \quad \text{HBA} \le 3, \quad \text{Rotatable Bonds} \le 3, \quad \text{TPSA} \le 60\ \text{Å}^2$$
  *(Note on historical literature: The original Congreve definition uses "$\le 3$", correcting the common slide typo "$< 3$").*
""")

    add_code(nb, """# Curate a toy fragment library and test Congreve's Rule of Three
sample_fragments = [
    {"name": "Phenol", "smiles": "c1ccccc1O"},
    {"name": "7-Azaindole", "smiles": "c1cc2c(nc1)[nH]cc2"},
    {"name": "Benzimidazole", "smiles": "c1ccc2[nH]cnc2c1"},
    {"name": "4-Aminopyridine", "smiles": "c1ccncc1N"},
    {"name": "2-Amino-1,3-thiazole", "smiles": "c1cscn1N"},
    {"name": "Quinoline core", "smiles": "c1ccc2ncccc2c1"},
    {"name": "Over-sized drug-like", "smiles": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"}  # Donepezil (MW > 300)
]

def check_rule_of_three(mol: Chem.Mol) -> dict:
    \"\"\"Check compliance with Congreve's Rule of Three.\"\"\"
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    passes = (
        mw <= 300.0 and
        clogp <= 3.0 and
        hbd <= 3 and
        hba <= 3 and
        rotb <= 3 and
        tpsa <= 60.0
    )
    return {
        "MW": round(mw, 2),
        "cLogP": round(clogp, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotB": rotb,
        "TPSA": round(tpsa, 2),
        "Passes_Ro3": passes
    }

results = []
for item in sample_fragments:
    m = Chem.MolFromSmiles(item["smiles"])
    info = check_rule_of_three(m)
    info["Name"] = item["name"]
    results.append(info)

df_ro3 = pd.DataFrame(results).set_index("Name")
df_ro3[["MW", "cLogP", "HBD", "HBA", "RotB", "TPSA", "Passes_Ro3"]]
""")

    add_md(nb, """## 3. Lead Optimization Metrics: Ligand Efficiency & Lipophilic Efficiency

When optimizing weak fragment hits ($K_D \sim 100\ \mu\text{M} - 1\ \text{mM}$) into high-affinity nanomolar leads, medicinal chemists track:

### 1. Ligand Efficiency (LE):
Free energy of binding normalized by the number of non-hydrogen (heavy) atoms:
$$LE = \frac{-\Delta G^\circ}{N_{\text{heavy}}} \approx \frac{1.37 \times \text{pIC}_{50}}{N_{\text{heavy}}} \quad (\text{kcal/mol per heavy atom})$$
- **Benchmark:** An efficient fragment has $LE \ge 0.30\ \text{kcal/mol/heavy atom}$. A drug candidate with $\text{MW} = 500$ ($N_{\text{heavy}} \approx 36$) and $K_i = 10\ \text{nM}$ ($\text{pIC}_{50} = 8.0$) achieves $LE \approx 10.96 / 36 \approx 0.30$.

### 2. Lipophilic Efficiency (LLE / LipE):
Measures potency gained specifically through directional interactions (H-bonds, electrostatic) rather than non-specific hydrophobic partitioning:
$$LLE = \text{pIC}_{50} - c\log P$$
- **Benchmark:** Good clinical leads target **$LLE \ge 5.0$**. If potency increases solely by adding greasy aromatic rings ($c\log P$ increases faster than $\text{pIC}_{50}$), $LLE$ drops, signaling poor metabolic stability.
""")

    add_code(nb, """# Calculate LE and LLE for simulated optimization series
optimization_series = pd.DataFrame([
    {"stage": "Fragment Hit", "smiles": "c1cc2c(nc1)[nH]cc2", "IC50_uM": 250.0, "cLogP": 1.20},
    {"stage": "Substituent Addition", "smiles": "Cc1cc2c(nc1)[nH]cc2Cl", "IC50_uM": 12.0, "cLogP": 1.85},
    {"stage": "Fragment Growing", "smiles": "Cc1cc2c(nc1)[nH]cc2-c1ccc(F)cc1", "IC50_uM": 0.45, "cLogP": 2.70},
    {"stage": "Optimized Clinical Candidate", "smiles": "Cc1cc2c(nc1)[nH]cc2-c1ccc(F)c(NS(=O)(=O)c2ccc(Cl)cc2)c1", "IC50_uM": 0.031, "cLogP": 3.40}
])

# Compute metrics
def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pIC50 = -np.log10(df["IC50_uM"] * 1e-6)
    heavy_atoms = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df["smiles"]]
    
    df["pIC50"] = np.round(pIC50, 2)
    df["N_heavy"] = heavy_atoms
    df["LE"] = np.round((1.37 * df["pIC50"]) / df["N_heavy"], 3)
    df["LLE"] = np.round(df["pIC50"] - df["cLogP"], 2)
    return df

df_opt = add_efficiency_metrics(optimization_series)
df_opt[["stage", "pIC50", "cLogP", "N_heavy", "LE", "LLE"]]
""")

    add_md(nb, """## 4. Successful FBDD Clinical Case Studies

### 1. Vemurafenib (PLX4032 / Zelboraf):
- **Target:** BRAF V600E mutant kinase (Metastatic Melanoma; FDA approved 2011).
- **Evolution:** Started from a tiny 7-azaindole fragment ($N_{\text{heavy}} = 9, IC_{50} > 200\ \mu\text{M}$ binding the kinase hinge region). Systematic structure-guided growing into the ATP-binding pocket yielded vemurafenib ($IC_{50} = 31\ \text{nM}, \text{MW} = 490$).

### 2. Venetoclax (Venclexta / ABT-199):
- **Target:** BCL-2 anti-apoptotic protein (Chronic Lymphocytic Leukemia; FDA approved 2016).
- **Evolution:** Overcame the formidable protein-protein interaction (PPI) groove via **fragment linking**: NMR screen detected two weak fragments binding adjacent sub-pockets ($K_D \approx 300\ \mu\text{M}$ and $K_D \approx 4\ \text{mM}$). Linking the two fragments with an acylsulfonamide linker produced a high-affinity sub-nanomolar inhibitor.
""")

    add_code(nb, """# Representing Vemurafenib evolution milestones
vemurafenib_milestones = [
    ("Starting Fragment (7-Azaindole)", "c1cc2c(nc1)[nH]cc2"),
    ("Core Lead (PLX4720)", "CS(=O)(=O)c1ccc(cc1)C(=O)c1c[nH]c2ncc(cc12)Cl"),
    ("Approved Drug (Vemurafenib)", "CCCS(=O)(=O)Nc1ccc(c(c1)C(=O)c2c[nH]c3ncc(cc23)Cl)F")
]

for name, smi in vemurafenib_milestones:
    m = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(m)
    clogp = Crippen.MolLogP(m)
    print(f"{name:35s} | SMILES: {smi[:35]}... | MW: {mw:.1f} | cLogP: {clogp:.2f}")
""")

    add_md(nb, """## 5. BRICS Fragment Decomposition & Synthetically Valid Assembly

**BRICS (Bioisosteric Rapid Intelligent Combinatorial Search)** (Degen et al., 2008) breaks molecular bonds according to **16 retrosynthetically accessible chemical reactions**:
- Bonds cleaved include amides, esters, alkyl-aryl amines, ethers, and aromatic C-C linkages.
- Cleaved attachment points are labeled with dummy atoms carrying isotope or environment tags (e.g. `[1*]`, `[5*]`), ensuring that subsequent fragment recombining follows real chemical reaction rules.
""")

    add_code(nb, """# Decompose Donepezil into BRICS fragments
donepezil_smi = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
mol_donepezil = Chem.MolFromSmiles(donepezil_smi)

# BRICS decomposition
fragments = BRICS.BRICSDecompose(mol_donepezil)
print(f"Donepezil decomposed into {len(fragments)} BRICS fragments:")
for idx, frag in enumerate(sorted(fragments), 1):
    print(f"  Frag {idx}: {frag}")
""")

    add_md(nb, """## 6. Guided Student Exercises & Self-Checks

---

### Exercise 1: Automated Rule-of-Three Filter
Implement the function `filter_rule_of_three(smiles_list: list) -> pd.DataFrame`:
- Computes MW, $c\log P$, HBD, HBA, RotB, and TPSA for each SMILES string.
- Evaluates Congreve's criteria:
  $$\text{MW} \le 300, \quad c\log P \le 3.0, \quad \text{HBD} \le 3, \quad \text{HBA} \le 3, \quad \text{RotB} \le 3, \quad \text{TPSA} \le 60.0$$
- Returns a DataFrame containing only the molecules that pass all 6 criteria.
""")

    add_code(nb, """# Exercise 1: Rule-of-Three Filter
def filter_rule_of_three(smiles_list: list) -> pd.DataFrame:
    \"\"\"Filter a list of SMILES strings for Congreve's Rule of Three compliance.\"\"\"
    # --- SOLUTION BEGIN ---
    passed_rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mw = Descriptors.MolWt(m)
        clogp = Crippen.MolLogP(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        rotb = Lipinski.NumRotatableBonds(m)
        tpsa = Descriptors.TPSA(m)
        
        if (mw <= 300.0 and clogp <= 3.0 and hbd <= 3 and 
            hba <= 3 and rotb <= 3 and tpsa <= 60.0):
            passed_rows.append({
                "smiles": smi,
                "MW": round(mw, 2),
                "cLogP": round(clogp, 2),
                "HBD": hbd,
                "HBA": hba,
                "RotB": rotb,
                "TPSA": round(tpsa, 2)
            })
    return pd.DataFrame(passed_rows)
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 1
test_smiles = [
    "c1ccccc1O",                 # Phenol (Passes)
    "c1cc2c(nc1)[nH]cc2",        # 7-Azaindole (Passes)
    "CCN(CC)CC",                 # Triethylamine (Passes)
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"  # Donepezil (Fails MW > 300)
]
df_passed = filter_rule_of_three(test_smiles)

assert isinstance(df_passed, pd.DataFrame), "Output must be a pandas DataFrame"
assert len(df_passed) == 3, f"Expected 3 passing fragments, got {len(df_passed)}"
assert "c1ccccc1O" in df_passed["smiles"].values
assert (df_passed["MW"] <= 300.0).all()
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Ligand Efficiency & Lipophilic Efficiency Calculator
Implement `compute_le_and_lle(smiles: str, ic50_nm: float) -> dict`:
- Calculates $\text{pIC}_{50} = 9.0 - \log_{10}(IC_{50}\ (\text{nM}))$.
- Obtains heavy atom count $N_{\text{heavy}}$ and $c\log P$ from RDKit.
- Calculates Ligand Efficiency: $LE = \frac{1.37 \times \text{pIC}_{50}}{N_{\text{heavy}}}$.
- Calculates Lipophilic Efficiency: $LLE = \text{pIC}_{50} - c\log P$.
- Returns dictionary with keys: `"pIC50"`, `"N_heavy"`, `"LE"`, `"LLE"`.
""")

    add_code(nb, """# Exercise 2: LE and LLE Calculator
def compute_le_and_lle(smiles: str, ic50_nm: float) -> dict:
    \"\"\"Compute pIC50, heavy atom count, LE, and LLE for a ligand.\"\"\"
    # --- SOLUTION BEGIN ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    pic50 = float(9.0 - math.log10(ic50_nm))
    n_heavy = mol.GetNumHeavyAtoms()
    clogp = float(Crippen.MolLogP(mol))
    
    le = float((1.37 * pic50) / n_heavy)
    lle = float(pic50 - clogp)
    
    return {
        "pIC50": round(pic50, 2),
        "N_heavy": n_heavy,
        "LE": round(le, 3),
        "LLE": round(lle, 2)
    }
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
res_eff = compute_le_and_lle("c1cc2c(nc1)[nH]cc2", 1000.0)  # 7-Azaindole at 1 uM (pIC50 = 6.0)
# N_heavy for 7-azaindole: 9 atoms
# LE = 1.37 * 6.0 / 9 = 0.913 kcal/mol/atom
assert abs(res_eff["pIC50"] - 6.0) < 1e-2, f"Expected pIC50 = 6.0, got {res_eff['pIC50']}"
assert res_eff["N_heavy"] == 9, f"Expected 9 heavy atoms, got {res_eff['N_heavy']}"
assert abs(res_eff["LE"] - 0.913) < 0.05, f"Expected LE ~ 0.91, got {res_eff['LE']}"
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: BRICS De Novo Library Assembly
Build a focused de novo candidate library by taking a core pharmacophore fragment and attaching functional groups:
Implement `generate_brics_library(core_smiles: str, substituent_smiles: list) -> list`:
- Attaches each substituent in `substituent_smiles` to the core.
- Converts to canonical SMILES and deduplicates.
- Returns a list of unique designed candidate SMILES.
""")

    add_code(nb, """# Exercise 3: BRICS Library Assembly
def generate_brics_library(core_smiles: str, substituent_smiles: list) -> list:
    \"\"\"Assemble a candidate library by attaching substituents to a scaffold core.\"\"\"
    # --- SOLUTION BEGIN ---
    unique_candidates = set()
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        return []

    # Using BRICS reaction SMARTS or direct coupling
    for sub in substituent_smiles:
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is not None:
            combo = Chem.CombineMols(core_mol, sub_mol)
            combo_smi = Chem.MolToSmiles(combo)
            # Create coupled analog
            ed = Chem.EditableMol(combo)
            # Connect atom 0 of core to atom 0 of substituent
            atom1 = 0
            atom2 = core_mol.GetNumAtoms()
            ed.AddBond(atom1, atom2, Chem.BondType.SINGLE)
            prod = ed.GetMol()
            try:
                Chem.SanitizeMol(prod)
                unique_candidates.add(Chem.MolToSmiles(prod))
            except Exception:
                pass
                
    return sorted(list(unique_candidates))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
core = "c1cc2c(nc1)[nH]cc2"  # 7-azaindole
substituents = ["c1ccccc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(C)cc1", "c1ccncc1", "c1ccc(OC)cc1"]
designed_lib = generate_brics_library(core, substituents)

print(f"Generated {len(designed_lib)} de novo candidates:")
for smi in designed_lib:
    print(f"  • {smi}")

assert len(designed_lib) >= 4, f"Expected at least 4 valid designs, got {len(designed_lib)}"
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """## Common Pitfalls

1. **Synthetic Inaccessibility**: Unconstrained de novo fragment growing often generates synthetically impossible or highly strained heterocyclic systems.
2. **Ignoring Ligand Efficiency**: High-affinity hits with high molecular weight often possess lower ligand efficiency than weaker, compact fragments.
3. **BRICS Disconnection Inconsistencies**: Assembling fragments without matching complementary synthetically valid synthons yields unchemical molecules.""")

    add_md(nb, """## Summary and Key Takeaways

- **FBDD Principles**: Fragment-based drug discovery leverages Rule of 3 (Ro3) chemical libraries to sample chemical space efficiently.
- **Efficiency Metrics**: Fragment growing, linking, and scaffold hopping are guided by Ligand Efficiency (LE) and Lipophilic Efficiency (LLE).
- **BRICS Synthesis**: Reaction rules ensure that de novo designs follow retro-synthetically accessible chemical transformation rules.""")

    add_code(nb, """# Final validation
print("✔ NB19 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb19_denovo_design_and_fbdd.ipynb",
        "solutions/nb19_denovo_design_and_fbdd.ipynb"
    )

if __name__ == "__main__":
    build_nb19()
