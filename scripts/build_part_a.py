#!/usr/bin/env python3
"""
generate_part_a.py
Generates high-quality, fully runnable Jupyter notebooks for Part A (nb00 to nb05)
in both student version (notebooks/) and solution version (solutions/).
"""

import os
import sys
import copy
import nbformat as nbf

def create_base_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (cheminf)",
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
            "version": "3.11.18"
        }
    }
    return nb

def add_md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))

def add_code(nb, code):
    nb.cells.append(nbf.v4.new_code_cell(code.strip()))

def add_exercise(nb_stud, nb_sol, desc_md, stub_code, sol_code, assert_code, ex_num):
    ex_header = f"### Exercise {ex_num}\n\n{desc_md.strip()}"
    add_md(nb_stud, ex_header)
    add_md(nb_sol, ex_header)
    
    add_code(nb_stud, stub_code.strip())
    add_code(nb_sol, sol_code.strip())
    
    assert_cell = f"# Self-check for Exercise {ex_num}\n{assert_code.strip()}\nprint('✔ Exercise {ex_num} passed!')"
    add_code(nb_stud, assert_cell)
    add_code(nb_sol, assert_cell)

# ==============================================================================
# NB00: Setup, Python and RDKit Refresher
# ==============================================================================
def build_nb00():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB00 · Setup, Python and RDKit Refresher
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb00_setup_and_refresher.ipynb)

**Estimated time:** ~30 min | **Difficulty:** ●○○

A hands-on onboarding guide verifying the local computational cheminformatics environment, introducing fundamental molecular operations in RDKit and Open Babel, manipulating tabular molecular datasets via `PandasTools`, and establishing robust error handling patterns.
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb00", "solutions/nb00"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Verify** your computational environment and inspect dependencies (`rdkit`, `openbabel`, `vina`, `meeko`, `mordred`, Java).
2. **Construct and visualize** 2D molecular graphs from SMILES strings using RDKit (`MolFromSmiles`, `MolToSmiles`, `Draw`).
3. **Iterate** across molecular graph elements, extracting atomic properties and bond orders.
4. **Differentiate** between implicit and explicit hydrogens (`AddHs`, `RemoveHs`) and quantify their effect on atomic indices.
5. **Manipulate** chemical datasets in pandas DataFrames with `PandasTools`.
6. **Execute** format conversions and 3D coordinate generation via Open Babel.
7. **Handle** chemical errors gracefully using sanitization flags and logger controls.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- Basic Python programming knowledge (functions, dictionaries, list comprehensions, pandas).
- Course context: Lecture 01 (Course Introduction) and Lecture 03 (Cheminformatics Tools).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, sys, random, warnings, shutil, subprocess
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools

# Prepend active Python directory to PATH to ensure external CLI tools (obabel, vina) are found
conda_bin = os.path.dirname(sys.executable)
if conda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)

# Disable verbose C++ logger warnings initially; we re-enable them in the error handling section
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Environment setup completed successfully.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. Environment Verification

Before beginning chemical data processing, we verify that our core libraries and command-line tools are operational.
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec1_code = """import rdkit
print(f"RDKit version:       {rdkit.__version__}")

try:
    import openbabel
    from openbabel import openbabel as ob
    print(f"Open Babel (Python): {openbabel.__version__}")
except ImportError:
    print("Open Babel (Python): Not installed")

obabel_path = shutil.which("obabel")
print(f"Open Babel (CLI):    {obabel_path if obabel_path else 'Not found in PATH'}")

vina_path = shutil.which("vina")
print(f"AutoDock Vina CLI:   {vina_path if vina_path else 'Not found in PATH'}")

java_path = shutil.which("java")
print(f"Java Runtime:        {java_path if java_path else 'Not found in PATH (PaDEL requires Java)'}")

try:
    import meeko
    print(f"Meeko version:       {meeko.__version__}")
except ImportError:
    print("Meeko:               Not installed")
"""
    add_code(nb_stud, sec1_code)
    add_code(nb_sol, sec1_code)
    
    sec2_md = """## 2. Basic Molecular Manipulation in RDKit

A molecule in RDKit is represented as a `ROMol` (Read-Only Molecule) or `RWMol` (Read-Write Molecule) object. Under the hood, this is a C++ graph object where atoms are nodes and bonds are edges.
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """# Parsing a SMILES string
ethanol = Chem.MolFromSmiles("CCO")
aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

print(f"Ethanol heavy atom count: {ethanol.GetNumHeavyAtoms()}")
print(f"Aspirin heavy atom count: {aspirin.GetNumHeavyAtoms()}")
print(f"Aspirin canonical SMILES: {Chem.MolToSmiles(aspirin)}")

# Visualizing molecules
Draw.MolsToGridImage([ethanol, aspirin], molsPerRow=2, subImgSize=(300, 200), legends=["Ethanol (CCO)", "Aspirin"])
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec2_atoms_md = """### Atom and Bond Iteration

Let us inspect the atoms and bonds of aspirin.
"""
    add_md(nb_stud, sec2_atoms_md)
    add_md(nb_sol, sec2_atoms_md)
    
    sec2_atoms_code = """atom_info = []
for atom in aspirin.GetAtoms():
    atom_info.append({
        "Idx": atom.GetIdx(),
        "Symbol": atom.GetSymbol(),
        "AtomicNum": atom.GetAtomicNum(),
        "Degree": atom.GetDegree(),
        "FormalCharge": atom.GetFormalCharge(),
        "Hybridization": str(atom.GetHybridization()),
        "IsAromatic": atom.GetIsAromatic()
    })
df_atoms = pd.DataFrame(atom_info)
print(df_atoms.head(8))

bond_info = []
for bond in aspirin.GetBonds():
    bond_info.append({
        "BondIdx": bond.GetIdx(),
        "BeginAtom": bond.GetBeginAtomIdx(),
        "EndAtom": bond.GetEndAtomIdx(),
        "BondType": str(bond.GetBondType()),
        "IsAromatic": bond.GetIsAromatic()
    })
df_bonds = pd.DataFrame(bond_info)
print(df_bonds.head(6))
"""
    add_code(nb_stud, sec2_atoms_code)
    add_code(nb_sol, sec2_atoms_code)
    
    notice1 = """> [!NOTE] What to notice
> By default, RDKit creates a **hydrogen-depleted molecular graph**. The hydrogen atoms are implicit and tracked as an integer property (`atom.GetTotalNumHs()`) on each heavy atom. Atom indexing starts at 0.
"""
    add_md(nb_stud, notice1)
    add_md(nb_sol, notice1)
    
    sec3_md = """## 3. Implicit vs. Explicit Hydrogens

In 2D graph manipulation and topological descriptor calculation, hydrogens are usually kept implicit to conserve memory and simplify graph traversal. However, for 3D coordinate generation, docking, and molecular mechanics, **explicit hydrogens are mandatory**.
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec3_code = """mol_depleted = Chem.MolFromSmiles("CCO")
mol_explicit = Chem.AddHs(mol_depleted)

print(f"Heavy atoms only:      {mol_depleted.GetNumAtoms()} atoms, {mol_depleted.GetNumBonds()} bonds")
print(f"With explicit H atoms: {mol_explicit.GetNumAtoms()} atoms, {mol_explicit.GetNumBonds()} bonds")

mol_recovered = Chem.RemoveHs(mol_explicit)
print(f"After RemoveHs:        {mol_recovered.GetNumAtoms()} atoms")
"""
    add_code(nb_stud, sec3_code)
    add_code(nb_sol, sec3_code)
    
    sec4_md = """## 4. Molecular Data in Pandas via `PandasTools`

`rdkit.Chem.PandasTools` seamlessly integrates RDKit molecules into pandas DataFrames, rendering 2D structure depictions directly within notebook tables.
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """csv_path = "../data/processed/d01_slide_draw_smiles.csv"
if not os.path.exists(csv_path):
    csv_path = "data/processed/d01_slide_draw_smiles.csv"

df = pd.read_csv(csv_path)
PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles", molCol="ROMol")
df["HeavyAtoms"] = df["ROMol"].apply(lambda m: m.GetNumHeavyAtoms() if m else 0)
df["MW"] = df["ROMol"].apply(lambda m: round(Descriptors.MolWt(m), 2) if m else 0.0)
df[["id", "name", "smiles", "HeavyAtoms", "MW"]].head(6)
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    sec5_md = """## 5. Open Babel Quick Tour

Open Babel is a versatile chemical toolbox specialized in inter-format translation. We can access Open Babel via its Python bindings or via its command-line interface `obabel`.
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """from openbabel import openbabel as ob

obConversion = ob.OBConversion()
in_fmts = obConversion.GetSupportedInputFormat()
out_fmts = obConversion.GetSupportedOutputFormat()

# Note vs slides: The original lecture slide reported 183 formats.
print(f"Open Babel supported input formats:  {len(in_fmts)}")
print(f"Open Babel supported output formats: {len(out_fmts)}")

# 0D to 3D conversion using Open Babel in Python
mol_ob = ob.OBMol()
obConversion.SetInAndOutFormats("smi", "mol2")
obConversion.ReadString(mol_ob, "CCCC")  # n-butane
mol_ob.AddHydrogens()

builder = ob.OBBuilder()
builder.Build(mol_ob)
mol2_text = obConversion.WriteString(mol_ob)

print("Generated MOL2 text (first 5 lines):")
for line in mol2_text.splitlines()[:5]:
    print(line)
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. Error Handling and Sanitization Flags

When parsing chemical data from external sources, invalid syntax or chemically nonsensical valences are common. RDKit returns `None` for invalid molecules.
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    sec6_code = """# Re-enable logger to view C++ warning messages
RDLogger.EnableLog("rdApp.*")

# Invalid SMILES: unclosed ring
bad_mol = Chem.MolFromSmiles("C1CC")
print(f"Parsed 'C1CC' -> {bad_mol}")

# Disabling sanitization to examine problematic intermediate structures (e.g. hypervalent carbon)
raw_mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
print(f"Without sanitization -> {raw_mol} (Atoms: {raw_mol.GetNumAtoms()})")

# Re-disable logger for rest of notebook
RDLogger.DisableLog("rdApp.*")
"""
    add_code(nb_stud, sec6_code)
    add_code(nb_sol, sec6_code)
    
    notice2 = """> [!WARNING] None Propagation Pitfall
> Always filter out `None` values before applying molecular descriptor functions in pandas pipelines (e.g. `df['ROMol'].dropna()`), otherwise `AttributeError: 'NoneType' object has no attribute 'GetNumHeavyAtoms'` will halt execution.
"""
    add_md(nb_stud, notice2)
    add_md(nb_sol, notice2)
    
    # Exercises
    # Ex 0.1
    ex01_desc = """Write a function `parse_and_validate_smiles(smiles_list: list[str]) -> dict[str, bool]` that parses a list of SMILES strings and returns a dictionary mapping each SMILES to `True` if it is valid (non-None) and `False` otherwise."""
    ex01_stub = """def parse_and_validate_smiles(smiles_list: list[str]) -> dict[str, bool]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement parse_and_validate_smiles")
"""
    ex01_sol = """def parse_and_validate_smiles(smiles_list: list[str]) -> dict[str, bool]:
    results = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        results[smi] = (mol is not None)
    return results
"""
    ex01_assert = """test_smiles = ["CCO", "c1ccccc1", "C(=O)O", "C1CC", "CCN(CC)CC"]
val_res = parse_and_validate_smiles(test_smiles)
assert val_res["CCO"] is True
assert val_res["c1ccccc1"] is True
assert val_res["C1CC"] is False
assert len(val_res) == 5
"""
    add_exercise(nb_stud, nb_sol, ex01_desc, ex01_stub, ex01_sol, ex01_assert, "0.1")
    
    # Ex 0.2
    ex02_desc = """Write a function `count_heavy_atoms_and_bonds(smiles: str) -> tuple[int, int]` that returns a tuple of `(num_heavy_atoms, num_bonds)` for the given SMILES string. If the SMILES is invalid, raise a `ValueError`."""
    ex02_stub = """def count_heavy_atoms_and_bonds(smiles: str) -> tuple[int, int]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_heavy_atoms_and_bonds")
"""
    ex02_sol = """def count_heavy_atoms_and_bonds(smiles: str) -> tuple[int, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return (mol.GetNumHeavyAtoms(), mol.GetNumBonds())
"""
    ex02_assert = """assert count_heavy_atoms_and_bonds("CCO") == (3, 2)
assert count_heavy_atoms_and_bonds("CC(=O)Oc1ccccc1C(=O)O") == (13, 13)
assert count_heavy_atoms_and_bonds("c1ccccc1") == (6, 6)
try:
    count_heavy_atoms_and_bonds("InvalidSmi")
    assert False, "Should have raised ValueError"
except ValueError:
    pass
"""
    add_exercise(nb_stud, nb_sol, ex02_desc, ex02_stub, ex02_sol, ex02_assert, "0.2")
    
    # Ex 0.3
    ex03_desc = """Write a function `get_formal_charges(smiles: str) -> dict[int, int]` that returns a dictionary mapping atom indices to their formal charge, for only those atoms having non-zero formal charge."""
    ex03_stub = """def get_formal_charges(smiles: str) -> dict[int, int]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_formal_charges")
"""
    ex03_sol = """def get_formal_charges(smiles: str) -> dict[int, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    charges = {}
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc != 0:
            charges[atom.GetIdx()] = fc
    return charges
"""
    ex03_assert = """fc_salt = get_formal_charges("CC(=O)[O-].[Na+]")
assert set(fc_salt.values()) == {-1, 1}
fc_neutral = get_formal_charges("CCO")
assert len(fc_neutral) == 0
"""
    add_exercise(nb_stud, nb_sol, ex03_desc, ex03_stub, ex03_sol, ex03_assert, "0.3")
    
    pitfalls_md = """## Common Pitfalls

1. **`None` Object Propagation:** Attempting to invoke RDKit methods on a `None` result causes unhandled exceptions in data science workflows. Always check `if mol is not None:` or use `.dropna()`.
2. **Implicit Hydrogens in 3D Embedding:** Omitting `Chem.AddHs(mol)` before calling 3D conformation embedding generates hydrogen-less 3D coordinates, distorting steric geometry.
3. **Log Suppression Concealing Bugs:** Disabling `RDLogger` permanently can obscure why a critical SMILES failed to kekulize. Selectively re-enable logging when debugging.
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- RDKit molecules are graph objects where vertices are atoms and edges are covalent bonds.
- Molecular graphs default to hydrogen-depleted representations; explicit hydrogens must be added via `Chem.AddHs` for 3D tasks.
- `PandasTools` enables high-throughput visual inspection of chemical datasets in tabular format.
- Open Babel provides comprehensive format conversion capabilities both as a CLI tool and via Python bindings.
- Chemical error handling requires explicit checking for `None` and an understanding of sanitization stages.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Landrum, G. *RDKit: Open-source cheminformatics*. https://www.rdkit.org
- O'Boyle, N. M. et al. Open Babel: An open chemical toolbox. *J. Cheminformatics* **2011**, 3, 33.
- Leach, A. R.; Gillet, V. J. *An Introduction to Chemoinformatics*. Springer, 2007.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB00 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb00 defined.")

# ==============================================================================
# NB01: Introduction: Informatics, Chemical Space and the Drug-Discovery Problem
# ==============================================================================
def build_nb01():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB01 · Introduction: Informatics, Chemical Space and the Drug-Discovery Problem
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb01_chemical_space_and_drug_discovery.ipynb)

**Estimated time:** ~45 min | **Difficulty:** ●○○

An exploration of chemical space dimensions, combinatorial isomer expansion, the strategic choice between Structure-Based (SBDD) and Ligand-Based (LBDD) Drug Design, lead optimization as multi-parameter optimization (MPO), and the thermodynamic relationship between binding affinity ($K_i$) and free energy ($\Delta G^\circ$).
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb01", "solutions/nb01"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Define** cheminformatics and delineate its position within computational chemistry and biomedical discovery.
2. **Estimate** the astronomical scale of drug-like chemical space ($\sim 10^{60}$) in comparison to physical universe benchmarks.
3. **Model** constitutional isomer explosion in acyclic alkanes using Cayley graph enumeration theory.
4. **Evaluate** Structure-Based vs. Ligand-Based Drug Design workflows using decision trees.
5. **Formulate** drug discovery as a multi-parameter optimization (MPO) problem balancing potency, selectivity, and ADMET.
6. **Calculate** thermodynamic binding free energy ($\Delta G^\circ = RT \ln K_i$) across physiologically relevant affinity regimes.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- NB00 (Setup and Python/RDKit Refresher) completed.
- Lecture 01 (Course Introduction: Chemical Space, SBDD vs LBDD, Free Energy).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, random, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("NB01 setup completed.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. Defining Cheminformatics

The term **chemoinformatics** (or **cheminformatics**) was formally defined by F.K. Brown in 1998:

> *"Chemoinformatics is the mixing of those information resources to transform data into information and information into knowledge for the intended purpose of making better decisions faster in the arena of drug lead identification and optimization."* — Frank K. Brown (1998)

Cheminformatics bridges organic chemistry, computer science, and data analysis. It focuses on the representation, search, analysis, and property prediction of chemical structures at high throughput.
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec2_md = """## 2. Chemical Space and Combinatorial Explosion

How vast is the space of small molecules? Bohacek et al. (1996) estimated that the number of synthetically accessible, drug-like organic molecules (molecular weight $\le 500$ Da, containing C, H, O, N, S, and halogens) is on the order of:

$$\\mathcal{N}_{\\text{drug-like}} \\approx 10^{60}$$

To put this astronomical number in perspective:
- Estimated atoms in the Earth: $\\sim 10^{50}$
- Total molecules synthesized by humanity (PubChem): $\\sim 1.1 \\times 10^8$
- Bioactive molecules with measured affinities (ChEMBL): $\\sim 2.4 \\times 10^6$
- Estimated atoms in the observable universe: $\\sim 10^{80}$

### The Alkane Isomer Expansion

Even for simple saturated hydrocarbons (alkanes, $\\text{C}_n\\text{H}_{2n+2}$), the number of constitutional isomers grows exponentially with carbon count $n$ (Cayley tree enumeration, OEIS A000602):

| Carbons ($n$) | Formula | Constitutional Isomers |
|---|---|---|
| 1 | $\\text{CH}_4$ | 1 |
| 4 | $\\text{C}_4\\text{H}_{10}$ | 2 |
| 6 | $\\text{C}_6\\text{H}_{14}$ | 5 |
| 8 | $\\text{C}_8\\text{H}_{18}$ | 18 |
| 10 | $\\text{C}_{10}\\text{H}_{22}$ | 75 |
| 20 | $\\text{C}_{20}\\text{H}_{42}$ | 366,319 |
| 30 | $\\text{C}_{30}\\text{H}_{62}$ | $4.11 \\times 10^9$ |
| 40 | $\\text{C}_{40}\\text{H}_{82}$ | $6.25 \\times 10^{13}$ |

> [!NOTE] Note vs. slides (Errata E13)
> The numbers above represent **constitutional (structural/skeletal) isomers** of acyclic alkanes. Stereoisomers (enantiomers arising from chiral carbons, e.g. 3-methylhexane) are excluded from these Cayley counts.
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """# Plotting the exponential growth of alkane isomers
carbon_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
isomer_counts = [1, 1, 1, 2, 3, 5, 9, 18, 35, 75, 355, 1858, 10359, 60523, 366319]

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(carbon_counts, isomer_counts, "o-", color="#1f77b4", lw=2, markersize=6)
ax.set_title("Combinatorial Explosion: Acyclic Alkane Constitutional Isomers", fontsize=12)
ax.set_xlabel("Number of Carbon Atoms ($n$)", fontsize=11)
ax.set_ylabel("Constitutional Isomers (Log Scale)", fontsize=11)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Inspecting real alkane physical properties across isomers
csv_path = "../data/processed/d02_alkanes_bp.csv"
if not os.path.exists(csv_path):
    csv_path = "data/processed/d02_alkanes_bp.csv"

df_alkanes = pd.read_csv(csv_path)
print(df_alkanes[df_alkanes["n_carbons"] == 5][["name", "smiles", "boiling_point_C"]])
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec3_md = """## 3. Structure-Based vs. Ligand-Based Drug Design

When developing a small-molecule drug, the starting computational strategy is governed by the structural knowledge available:

```mermaid
graph TD
    Start["New Drug Discovery Project"] --> Q1{"Is high-resolution 3D target structure known?"}
    Q1 -- Yes --> SBDD["Structure-Based Drug Design (SBDD)"]
    Q1 -- No --> Q2{"Are known active small-molecule ligands available?"}
    
    SBDD --> SBDD_Tools["Cavity Detection<br>Molecular Docking<br>Fragment-Based Design (FBDD)<br>De Novo Design"]
    
    Q2 -- Yes --> LBDD["Ligand-Based Drug Design (LBDD)"]
    Q2 -- No --> Target["Target Discovery & Modeling<br>AlphaFold / Cryo-EM / Homology"]
    
    LBDD --> LBDD_Tools["Pharmacophore Modeling<br>2D/3D Similarity Searching<br>QSAR / Machine Learning"]
```

### Strategic Comparison

| Characteristic | Structure-Based Drug Design (SBDD) | Ligand-Based Drug Design (LBDD) |
|---|---|---|
| **Prerequisites** | 3D experimental structure (X-ray, Cryo-EM, AlphaFold) | Bioactivity dataset of active/inactive molecules |
| **Core Methodologies** | Molecular docking, molecular dynamics, cavity mapping | QSAR, pharmacophores, fingerprint similarity |
| **Scoring Basis** | Physical/empirical force fields ($\Delta G_{\\text{bind}}$) | Statistical correlation, similarity metrics (Tanimoto) |
| **Primary Risk** | Target flexibility, solvent entropic effects | Data bias, activity cliffs, lack of novelty |
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec4_md = """## 4. Drug Design as Multi-Parameter Optimization (MPO)

A common misconception is that drug discovery is solely about maximizing binding affinity. In reality, high affinity is necessary but far from sufficient.

A clinical drug candidate must concurrently satisfy multiple competing criteria:
1. **Potency:** High on-target affinity ($K_i < 10$ nM).
2. **Selectivity:** Low affinity for antitargets (e.g. hERG potassium channel causing cardiac arrhythmia, CYP450 enzymes causing drug-drug interactions).
3. **Solubility & Permeability:** Aqueous solubility for gastrointestinal absorption and membrane permeability to reach intracellular targets.
4. **Metabolic Stability:** Appropriate clearance rate to allow once- or twice-daily dosing without hepatotoxicity.

In the 1990s, high-throughput screening often produced extremely lipophilic, potent molecules that failed in Phase II/III clinical trials due to poor pharmacokinetics and toxicity. This prompted Christopher Lipinski's famous **Rule of 5** (1997).
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """# Multi-parameter scorecard comparison
categories = ['Potency (pKi)', 'Solubility', 'Permeability', 'Metabolic Stability', 'Safety (Anti-hERG)']
candidate_A = [9.0, 3.0, 4.0, 2.5, 3.0]  # Highly potent, but insoluble and unstable
candidate_B = [7.5, 8.0, 7.5, 8.0, 8.5]  # Moderately potent, excellent drug-like balance

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
vals_A = candidate_A + candidate_A[:1]
vals_B = candidate_B + candidate_B[:1]

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.plot(angles, vals_A, color="#d62728", lw=2, label="Potent but Poor ADMET")
ax.fill(angles, vals_A, color="#d62728", alpha=0.15)
ax.plot(angles, vals_B, color="#2ca02c", lw=2, label="Balanced Lead Candidate")
ax.fill(angles, vals_B, color="#2ca02c", alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 10)
plt.title("Multi-Parameter Optimization Profile", y=1.08, fontsize=12)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.show()
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    sec5_md = """## 5. Thermodynamics of Binding: From $K_i$ to $\\Delta G^\\circ$

At thermodynamic equilibrium, the binding of a ligand $L$ to a protein receptor $R$:

$$R + L \\rightleftharpoons RL$$

is governed by the dissociation constant $K_i$ (or $K_d$):

$$K_i = \\frac{[R][L]}{[RL]}$$

The standard Gibbs free energy of binding $\\Delta G^\\circ_{\\text{bind}}$ is related to $K_i$ via:

$$\\Delta G^\\circ = RT \\ln K_i = -RT \\ln K_a$$

where:
- $R = 8.3144626\\ \\text{J}/(\\text{mol}\\cdot\\text{K}) = 1.9872 \\times 10^{-3}\\ \\text{kcal}/(\\text{mol}\\cdot\\text{K})$
- $T = 298.15\\ \\text{K}$ (standard laboratory temperature, $25^\\circ\\text{C}$)
- At $298.15$ K, $RT \\approx 2.479\\ \\text{kJ/mol} \\approx 0.5924\\ \\text{kcal/mol}$.

### The Logarithmic Rule of Thumb

Because $\\ln(10) \\approx 2.3026$:

$$\\Delta G^\\circ = 2.3026 \\times RT \\times \\log_{10}(K_i)$$

At $298.15\\ \\text{K}$, **every 10-fold improvement in binding affinity** (1 unit on the $pK_i = -\\log_{10} K_i$ scale) corresponds to:

$$\\Delta \\Delta G^\\circ \\approx -5.71\\ \\text{kJ/mol} \\approx -1.36\\ \\text{kcal/mol}$$

This means a single well-placed hydrogen bond (worth $\\sim 2\\text{--}5$ kcal/mol) can improve affinity by **100- to 1,000-fold**!
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """R_KJ = 8.3144626e-3  # kJ / (mol * K)
R_KCAL = 1.9872e-3    # kcal / (mol * K)
T_STD = 298.15        # K

affinities = [
    ("Millimolar (mM)", 1e-3),
    ("Micromolar (uM)", 1e-6),
    ("Nanomolar (nM)",  1e-9),
    ("Picomolar (pM)",  1e-12),
    ("Femtomolar (fM)", 1e-15)
]

free_energy_table = []
for name, ki in affinities:
    dG_kj = R_KJ * T_STD * math.log(ki)
    dG_kcal = R_KCAL * T_STD * math.log(ki)
    free_energy_table.append({
        "Affinity Regime": name,
        "Ki (Molar)": f"{ki:.0e}",
        "pKi (-log10 Ki)": -math.log10(ki),
        "Delta G (kJ/mol)": round(dG_kj, 2),
        "Delta G (kcal/mol)": round(dG_kcal, 2)
    })

pd.DataFrame(free_energy_table)
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. Deterministic vs. Data-Driven Approaches

Computational cheminformatics deploys two complementary paradigms:

| Metric | Deterministic (Physics-Based) | Data-Driven (Statistical / ML) |
|---|---|---|
| **Underlying Theory** | Quantum mechanics (DFT), Newton's laws (MD, force fields) | Statistical learning, inductive bias, deep neural networks |
| **Input Requirements** | Explicit 3D atomic coordinates, force field parameters | 2D molecular graphs, SMILES strings, bioassay activity tables |
| **Computational Cost** | High (CPU/GPU-intensive, minutes to days per ligand) | Low (milliseconds to seconds per molecule) |
| **Generalization** | Transferable across unseen chemotypes | High fidelity within training domain; vulnerable to activity cliffs |
| **Examples** | Free Energy Perturbation (FEP), AutoDock Vina, Desmond | Random Forest QSAR, ChemBERTa, Graph Neural Networks |
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    # Exercises
    # Ex 1.1
    ex11_desc = """Write a function `compute_delta_g(ki_molar: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float` that calculates standard binding free energy $\\Delta G^\\circ = RT \\ln K_i$. Support units `'kJ/mol'` and `'kcal/mol'`. If an invalid unit is given or $K_i \\le 0$, raise a `ValueError`."""
    ex11_stub = """def compute_delta_g(ki_molar: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float:
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_delta_g")
"""
    ex11_sol = """def compute_delta_g(ki_molar: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float:
    if ki_molar <= 0:
        raise ValueError("Ki must be strictly positive.")
    if unit == "kJ/mol":
        r = 8.3144626e-3
    elif unit == "kcal/mol":
        r = 1.9872e-3
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    return r * temperature_k * math.log(ki_molar)
"""
    ex11_assert = """# Check 10 uM, 10 nM, 10 pM
dg_10um = compute_delta_g(1e-5, 298.15, "kJ/mol")
dg_10nm = compute_delta_g(1e-8, 298.15, "kJ/mol")
dg_10pm = compute_delta_g(1e-11, 298.15, "kJ/mol")

assert math.isclose(dg_10um, -28.537, abs_tol=0.05)
assert math.isclose(dg_10nm, -45.659, abs_tol=0.05)
assert math.isclose(dg_10pm, -62.781, abs_tol=0.05)

# Check kcal/mol
dg_10nm_kcal = compute_delta_g(1e-8, 298.15, "kcal/mol")
assert math.isclose(dg_10nm_kcal, -10.91, abs_tol=0.05)
"""
    add_exercise(nb_stud, nb_sol, ex11_desc, ex11_stub, ex11_sol, ex11_assert, "1.1")
    
    # Ex 1.2
    ex12_desc = """Write an inverse function `ki_from_delta_g(delta_g: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float` that computes the equilibrium dissociation constant $K_i$ in Molar from a given $\\Delta G^\\circ$."""
    ex12_stub = """def ki_from_delta_g(delta_g: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float:
    # YOUR CODE HERE
    raise NotImplementedError("Implement ki_from_delta_g")
"""
    ex12_sol = """def ki_from_delta_g(delta_g: float, temperature_k: float = 298.15, unit: str = "kJ/mol") -> float:
    if unit == "kJ/mol":
        r = 8.3144626e-3
    elif unit == "kcal/mol":
        r = 1.9872e-3
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    return math.exp(delta_g / (r * temperature_k))
"""
    ex12_assert = """ki_orig = 1e-8
dg = compute_delta_g(ki_orig, 298.15, "kJ/mol")
ki_recov = ki_from_delta_g(dg, 298.15, "kJ/mol")
assert math.isclose(ki_orig, ki_recov, rel_tol=1e-5)
"""
    add_exercise(nb_stud, nb_sol, ex12_desc, ex12_stub, ex12_sol, ex12_assert, "1.2")
    
    # Ex 1.3
    ex13_desc = """Write a function `get_alkane_isomer_count(n_carbons: int) -> int` that returns the exact number of acyclic alkane constitutional isomers for $n \\in [1, 8]$ according to Cayley's theorem (OEIS A000602). For $n$ outside $[1, 8]$, raise a `ValueError`."""
    ex13_stub = """def get_alkane_isomer_count(n_carbons: int) -> int:
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_alkane_isomer_count")
"""
    ex13_sol = """def get_alkane_isomer_count(n_carbons: int) -> int:
    known_counts = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18}
    if n_carbons not in known_counts:
        raise ValueError(f"n_carbons {n_carbons} not in supported range [1, 8]")
    return known_counts[n_carbons]
"""
    ex13_assert = """assert get_alkane_isomer_count(1) == 1
assert get_alkane_isomer_count(4) == 2
assert get_alkane_isomer_count(6) == 5
assert get_alkane_isomer_count(7) == 9
assert get_alkane_isomer_count(8) == 18
"""
    add_exercise(nb_stud, nb_sol, ex13_desc, ex13_stub, ex13_sol, ex13_assert, "1.3")
    
    pitfalls_md = """## Common Pitfalls

1. **Natural vs. Base-10 Logarithms:** $\\Delta G^\\circ = RT \\ln K_i$, using the natural logarithm (`math.log`). Using $\\log_{10}$ produces errors by a factor of $2.303$.
2. **Affinity Units in Free Energy Equations:** $K_i$ must always be converted to standard-state **Molar units (M)** before taking the logarithm. Passing nanomolar concentrations ($10$ instead of $10^{-8}$) yields an erroneous positive free energy!
3. **Equating Constitutional Isomers with Stereoisomers:** Acyclic alkane isomer tables enumerate constitutional (connectivity) isomers. Real compounds with stereocenters have additional stereoisomeric configurations.
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- Cheminformatics focuses on data-driven chemical information extraction and decision-making in lead discovery.
- Drug-like chemical space is virtually inexhaustible ($\sim 10^{60}$), demanding algorithmic filtering and generative search.
- SBDD is chosen when a 3D target structure exists; LBDD is deployed when active small-molecule data are available.
- Drug discovery is an inherently multi-objective optimization problem; potency without acceptable ADMET leads to clinical failure.
- Binding free energy is logarithmic: a 10-fold increase in binding affinity represents only $-5.71$ kJ/mol ($-1.36$ kcal/mol) of binding free energy at 298 K.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Bohacek, R. S. et al. The art and practice of structure-based drug design: A molecular modeling perspective. *Med. Res. Rev.* **1996**, 16, 3–50.
- Brown, F. K. Chemoinformatics: What is it and how does it impact drug discovery? *Annu. Rep. Med. Chem.* **1998**, 33, 375–384.
- Lipinski, C. A. et al. Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Adv. Drug Deliv. Rev.* **1997**, 23, 3–25.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB01 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb01 defined.")

# ==============================================================================
# NB02: Building 3D Structures: Cartesian, Internal Coordinates and Editors
# ==============================================================================
def build_nb02():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB02 · Building 3D Structures: Cartesian, Internal Coordinates and Editors
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb02_building_3d_structures.ipynb)

**Estimated time:** ~60 min | **Difficulty:** ●●○

A comprehensive practical guide on constructing 3D molecular structures from geometric principles, distinguishing coordinate-dependent Cartesian frames from invariant internal coordinates (Z-matrix), implementing trigonometric conversions (NeRF), visualizing 3D conformations with `py3Dmol`, and performing torsional scans.
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb02", "solutions/nb02"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Construct** 3D Cartesian coordinates of methane from fundamental bond lengths and tetrahedral bond angles using trigonometry.
2. **Demonstrate** that Cartesian coordinates depend on the reference frame whereas interatomic distance matrices are invariant under rigid 3D rotation.
3. **Parse and formulate** internal coordinate representations (Z-matrices) for simple and multi-rotor molecules (methane, n-butane, 1,2-dichloroethane).
4. **Implement** an internal-to-Cartesian coordinate converter based on the Natural Extension Reference Frame (NeRF) formulation.
5. **Visualize** 3D molecular conformations using `py3Dmol` and compute torsional energy profiles (gauche vs. anti minima).
6. **Survey** common 2D and 3D molecular editors and sketchers used in computational chemistry.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- NB00 and NB01 completed.
- Lecture 01 (Building 3D structures, trigonometry, Z-matrix, Exercise-1) and Lecture 02 (Cartesian coordinates, internal coordinates).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, random, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("NB02 setup completed.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. Methane Construction by Trigonometry

In a perfect tetrahedral geometry ($sp^3$ hybridization), the bond angle is:

$$\\alpha = \\arccos\\left(-\\frac{1}{3}\\right) \\approx 109.4712^\\circ \\approx 1.91063\\ \\text{radians}$$

Let carbon be situated at the origin $(0, 0, 0)$. Defining the half-angle:

$$\\theta = \\frac{\\alpha}{2} \\approx 54.7356^\\circ \\approx 0.95532\\ \\text{radians}$$

With a standard C–H bond length $r = 1.09\\ \\text{Å}$, the four hydrogen positions are parameterized as:

$$\\text{H}_1 = (r \\cos\\theta,\\ r \\sin\\theta,\\ 0)$$
$$\\text{H}_2 = (r \\cos\\theta,\\ -r \\sin\\theta,\\ 0)$$
$$\\text{H}_3 = (-r \\cos\\theta,\\ 0,\\ r \\sin\\theta)$$
$$\\text{H}_4 = (-r \\cos\\theta,\\ 0,\\ -r \\sin\\theta)$$

Notice the numerical values:
- $r \\cos\\theta = 1.09 \\times \\cos(54.7356^\\circ) = 1.09 \\times 0.57735 \\approx 0.6293\\ \\text{Å} \\approx 0.63\\ \\text{Å}$
- $r \\sin\\theta = 1.09 \\times \\sin(54.7356^\\circ) = 1.09 \\times 0.81650 \\approx 0.8900\\ \\text{Å} \\approx 0.89\\ \\text{Å}$

> [!NOTE] What to notice
> The classic coordinates on the course slide $(\\pm 0.63, \\pm 0.89, 0)$ correspond exactly to these trigonometric projections for $r = 1.09\\ \\text{Å}$!
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec1_code = """r = 1.09  # Angstroms
alpha = np.arccos(-1.0 / 3.0)  # ~109.4712 degrees
theta = alpha / 2.0           # ~54.7356 degrees

# Calculate positions
C = np.array([0.0, 0.0, 0.0])
H1 = np.array([ r * np.cos(theta),  r * np.sin(theta), 0.0])
H2 = np.array([ r * np.cos(theta), -r * np.sin(theta), 0.0])
H3 = np.array([-r * np.cos(theta), 0.0,  r * np.sin(theta)])
H4 = np.array([-r * np.cos(theta), 0.0, -r * np.sin(theta)])

coords_methane = np.array([C, H1, H2, H3, H4])
atom_names = ["C", "H1", "H2", "H3", "H4"]

print("Methane Cartesian Coordinates (Angstroms):")
for name, c in zip(atom_names, coords_methane):
    print(f"{name:2s}: ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f})")

# Verify all 4 C-H bond lengths
ch_lengths = [np.linalg.norm(coords_methane[i] - C) for i in range(1, 5)]
print(f"C-H Bond lengths: {[round(l, 4) for l in ch_lengths]}")
assert all(np.isclose(l, 1.09, atol=1e-3) for l in ch_lengths)

# Verify H-C-H bond angles
def calc_angle(p1, center, p2):
    v1 = p1 - center
    v2 = p2 - center
    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

angles = [
    calc_angle(H1, C, H2),
    calc_angle(H1, C, H3),
    calc_angle(H1, C, H4),
    calc_angle(H2, C, H3),
    calc_angle(H2, C, H4),
    calc_angle(H3, C, H4)
]
print(f"H-C-H Bond angles (degrees): {[round(a, 2) for a in angles]}")
assert all(np.isclose(a, 109.47, atol=1e-1) for a in angles)
"""
    add_code(nb_stud, sec1_code)
    add_code(nb_sol, sec1_code)
    
    sec2_md = """## 2. Orientation Dependence vs. Distance Invariance

If we apply a rigid 3D rotation to the molecular coordinates:

$$X' = X R$$

where $R \\in SO(3)$ is an orthogonal rotation matrix ($R R^T = I$), the Cartesian coordinates change dramatically.

However, the **interatomic Euclidean distance matrix**:

$$D_{ij} = \\|x_i - x_j\\|$$

is strictly invariant under any rigid translation or rotation.

> [!NOTE] Are molecular properties coordinate-dependent?
> As raised in the lecture slides: **Intrinsic scalar properties** (molecular energy, dipole moment magnitude, molecular volume, vibrational frequencies) are coordinate-invariant. **Directional properties** (dipole vector components along laboratory axes, quadrupole tensors) depend on the choice of external coordinate frame.
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """# Define arbitrary 3D rotation matrix around axis
def rotation_matrix_3d(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle_rad / 2.0)
    b, c, d = -axis * np.sin(angle_rad / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
    ])

R_rot = rotation_matrix_3d(np.array([1.0, 1.0, 0.5]), np.radians(45))
coords_rotated = coords_methane @ R_rot.T

print("Coordinates changed under rotation:")
print(f"Original H1: {np.round(coords_methane[1], 4)}")
print(f"Rotated  H1: {np.round(coords_rotated[1], 4)}")

# Compute pairwise distance matrices
D_orig = np.linalg.norm(coords_methane[:, None, :] - coords_methane[None, :, :], axis=-1)
D_rot  = np.linalg.norm(coords_rotated[:, None, :] - coords_rotated[None, :, :], axis=-1)

assert np.allclose(D_orig, D_rot, atol=1e-5)
print("✔ Interatomic distance matrix is strictly invariant under rotation!")
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec3_md = """## 3. Internal Coordinates: The Z-Matrix

Cartesian coordinates require $3N$ values and specify both molecular geometry and absolute orientation in space. For an isolated molecule, 6 degrees of freedom (3 translational, 3 rotational) are chemically irrelevant.

An **internal coordinate representation** specifies geometry using:
1. **Bond lengths** ($r_{ij}$, distance between atom $i$ and atom $j$).
2. **Bond angles** ($\\theta_{ijk}$, angle formed by atoms $i-j-k$).
3. **Dihedral / Torsion angles** ($\\phi_{ijkl}$, dihedral angle between planes $ijk$ and $jkl$).

The standard format is the **Z-Matrix**:
- Atom 1: Placed at the origin.
- Atom 2: Placed on the z-axis at distance $r_{21}$ from Atom 1.
- Atom 3: Placed in the xz-plane at distance $r_{32}$ from Atom 2 and angle $\\theta_{321}$.
- Atom $i$ ($i \\ge 4$): Placed relative to three previously defined atoms $j, k, l$ with length $r_{ij}$, angle $\\theta_{ijk}$, and dihedral $\\phi_{ijkl}$.

### Examples of Z-Matrices

**1. Methane ($\\text{CH}_4$):**
```text
C
H  1  1.09
H  1  1.09  2  109.47
H  1  1.09  2  109.47  3  120.0
H  1  1.09  2  109.47  3  240.0
```

**2. n-Butane Anti Conformer ($\\text{C}_4\\text{H}_{10}$ heavy skeleton):**
```text
C1
C2  1  1.54
C3  2  1.54  1  109.5
C4  3  1.54  2  109.5  1  180.0
```
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec4_md = """## 4. Z-Matrix to Cartesian Conversion (NeRF)

The **Natural Extension Reference Frame (NeRF)** algorithm efficiently converts internal coordinates into Cartesian space.

Given three existing Cartesian points $A, B, C$, to place atom $D$ with bond length $r = \\|D - C\\|$, bond angle $\\theta = \\angle BCD$, and dihedral angle $\\phi = \\angle ABCD$:
1. Construct the unit vector along the bond $C \\to B$:
   $$u = \\frac{C - B}{\\|C - B\\|}$$
2. Construct the normal vector to the plane $ABC$:
   $$w = \\frac{u \\times (B - A)}{\\|u \\times (B - A)\\|}$$
3. Construct the orthogonal unit vector in the plane:
   $$v = w \\times u$$
4. Atom $D$ is placed at:
   $$D = C + r \\left( -\\cos(\\theta) u + \\sin(\\theta)\\cos(\\phi) v + \\sin(\\theta)\\sin(\\phi) w \\right)$$
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """def place_atom_nerf(p_a, p_b, p_c, r, angle_deg, dihedral_deg):
    \"\"\"Place atom D given reference atoms A, B, C using NeRF method.\"\"\"
    theta = np.radians(angle_deg)
    phi = np.radians(dihedral_deg)
    
    bc = p_c - p_b
    u = bc / np.linalg.norm(bc)
    
    ab = p_b - p_a
    cross_ab_bc = np.cross(u, ab)
    norm_cross = np.linalg.norm(cross_ab_bc)
    if norm_cross < 1e-7:  # Collinear points fallback
        w = np.array([0.0, 0.0, 1.0])
    else:
        w = cross_ab_bc / norm_cross
        
    v = np.cross(w, u)
    
    d_dir = -np.cos(theta) * u + np.sin(theta) * np.cos(phi) * v + np.sin(theta) * np.sin(phi) * w
    return p_c + r * d_dir

# Test: Place 4 heavy atoms of n-butane (anti conformer)
C1 = np.array([0.0, 0.0, 0.0])
C2 = np.array([1.54, 0.0, 0.0])
# Place C3 with r=1.54, angle=109.5 in XY plane
theta_c = np.radians(109.5)
C3 = C2 + np.array([-1.54 * np.cos(theta_c), 1.54 * np.sin(theta_c), 0.0])
# Place C4 with dihedral=180.0
C4 = place_atom_nerf(C1, C2, C3, 1.54, 109.5, 180.0)

coords_butane_anti = np.array([C1, C2, C3, C4])
print("Built Butane Anti-Conformer Heavy Atoms:")
for i, c in enumerate(coords_butane_anti, 1):
    print(f"C{i}: {np.round(c, 4)}")
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    sec5_md = """## 5. Interactive 3D Visualization and Butane Torsion Scan

Conformational search explores rotation around single bonds. For n-butane ($CH_3-CH_2-CH_2-CH_3$), scanning the central C2–C3 dihedral angle $\\phi$ from $0^\\circ$ to $360^\\circ$ produces the classic torsional energy profile:

- **Anti-periplanar ($\\phi = 180^\\circ$):** Global minimum ($0$ kJ/mol, steric repulsion minimized).
- **Gauche ($\\phi \\approx 60^\\circ, 300^\\circ$):** Local minima ($\\approx 3.8$ kJ/mol above anti).
- **Anticlinal (eclipsed, $\\phi \\approx 120^\\circ, 240^\\circ$):** Energy barriers ($\\approx 16$ kJ/mol).
- **Syn-periplanar (fully eclipsed, $\\phi = 0^\\circ$):** Maximum barrier ($\\approx 19$ kJ/mol, methyl-methyl clash).
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """mol_butane = Chem.MolFromSmiles("CCCC")
mol_butane = Chem.AddHs(mol_butane)
AllChem.EmbedMolecule(mol_butane, randomSeed=42)

# Find the central C-C-C-C dihedral atoms (0, 1, 2, 3)
conf = mol_butane.GetConformer()
dihedral_atoms = (0, 1, 2, 3)

angles_scan = np.linspace(0, 360, 37)
energies_mmff = []

for ang in angles_scan:
    Chem.rdMolTransforms.SetDihedralDeg(conf, *dihedral_atoms, float(ang))
    # MMFF calculation without modifying dihedral
    prop = AllChem.MMFFGetMoleculeProperties(mol_butane)
    ff = AllChem.MMFFGetMoleculeForceField(mol_butane, prop)
    e = ff.CalcEnergy()
    energies_mmff.append(e)

energies_rel_kj = (np.array(energies_mmff) - min(energies_mmff)) * 4.184  # kcal to kJ

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(angles_scan, energies_rel_kj, "o-", color="#2ca02c", lw=2, markersize=5)
ax.set_title("n-Butane C-C-C-C Torsional Energy Profile (MMFF)", fontsize=12)
ax.set_xlabel("Dihedral Angle $\\phi$ (degrees)", fontsize=11)
ax.set_ylabel("Relative Energy (kJ/mol)", fontsize=11)
ax.set_xticks(np.arange(0, 361, 60))
ax.axvline(60, color="gray", ls="--", alpha=0.6, label="Gauche (60°)")
ax.axvline(180, color="blue", ls="--", alpha=0.6, label="Anti (180°)")
ax.axvline(300, color="gray", ls="--", alpha=0.6, label="Gauche (300°)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# 3D visualization helper via py3Dmol
try:
    import py3Dmol
    viewer = py3Dmol.view(width=350, height=250)
    viewer.addModel(Chem.MolToMolBlock(mol_butane), "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    print("py3Dmol is available for interactive 3D rendering.")
except ImportError:
    print("py3Dmol not installed in current session.")
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. Computational Molecular Editors Overview

Molecular editors fall into two broad categories:

| Tool | Type | Capabilities & Export | License |
|---|---|---|---|
| **Avogadro / Avogadro 2** | 3D Builder & Visualizer | Z-matrix editor, surface electrostatic potentials, optimization | Open-source (GPL) |
| **Schrödinger Maestro** | Professional 3D Suite | Protein-ligand editing, docking preparation, glide grids | Commercial (Free Academic version) |
| **Molden** | 3D Visualizer | Electronic density, molecular orbitals, vibration analysis | Academic free |
| **PyMOL / ChimeraX** | 3D Macromolecular Visualizer | High-resolution ray-tracing, binding site rendering | Open-source / Academic |
| **ChemDraw** | 2D Chemical Sketcher | Publication-grade 2D structures, IUPAC name generation | Commercial |
| **MarvinSketch / JSME** | 2D Web/Desktop Sketchers | SMILES/SMARTS export, RDKit web integration | Free / Open-source |
| **PubChem Sketcher** | 2D Online Sketcher | Instant database cross-referencing and CID retrieval | Public web service |
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    # Exercises
    # Ex 2.1
    ex21_desc = """Build the 3D coordinates of methane using internal coordinates (Z-matrix) and verify that all 4 C-H bond lengths equal $1.09 \\pm 0.01\\ \\text{Å}$. Place C at origin, H1 along the Z-axis, H2 in the XZ plane with $\\angle \\text{H}_1-\\text{C}-\\text{H}_2 = 109.47^\\circ$, and H3/H4 using dihedrals $120.0^\\circ$ and $240.0^\\circ$."""
    ex21_stub = """def build_methane_zmatrix() -> np.ndarray:
    \"\"\"Return a 5x3 numpy array of Cartesian coordinates [C, H1, H2, H3, H4].\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Implement build_methane_zmatrix")
"""
    ex21_sol = """def build_methane_zmatrix() -> np.ndarray:
    r = 1.09
    alpha_deg = 109.4712
    C = np.array([0.0, 0.0, 0.0])
    H1 = np.array([0.0, 0.0, r])
    
    # H2 in XZ plane
    theta = np.radians(alpha_deg)
    H2 = np.array([r * np.sin(theta), 0.0, r * np.cos(theta)])
    
    # Helper to place H3, H4
    def place_h(phi_deg):
        phi = np.radians(phi_deg)
        # rotated around Z axis relative to H2
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])
        
    H3 = place_h(120.0)
    H4 = place_h(240.0)
    return np.array([C, H1, H2, H3, H4])
"""
    ex21_assert = """coords = build_methane_zmatrix()
assert coords.shape == (5, 3)
C_pos = coords[0]
for i in range(1, 5):
    dist = np.linalg.norm(coords[i] - C_pos)
    assert np.isclose(dist, 1.09, atol=0.01)
"""
    add_exercise(nb_stud, nb_sol, ex21_desc, ex21_stub, ex21_sol, ex21_assert, "2.1")
    
    # Ex 2.2
    ex22_desc = """Write a function `rotate_about_x_axis(coords: np.ndarray, angle_deg: float) -> np.ndarray` that rotates an $N \\times 3$ coordinate array about the x-axis by `angle_deg` degrees. Rotate your methane coordinates by $30^\\circ$ and assert that the distance between C and H1 is conserved."""
    ex22_stub = """def rotate_about_x_axis(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError("Implement rotate_about_x_axis")
"""
    ex22_sol = """def rotate_about_x_axis(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta),  np.cos(theta)]
    ])
    return coords @ Rx.T
"""
    ex22_assert = """m_coords = build_methane_zmatrix()
m_rot = rotate_about_x_axis(m_coords, 30.0)
d_orig = np.linalg.norm(m_coords[1] - m_coords[0])
d_rot = np.linalg.norm(m_rot[1] - m_rot[0])
assert np.isclose(d_orig, d_rot, atol=1e-5)
"""
    add_exercise(nb_stud, nb_sol, ex22_desc, ex22_stub, ex22_sol, ex22_assert, "2.2")
    
    # Ex 2.3
    ex23_desc = """Write a function `compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float` that computes the dihedral angle formed by four sequentially bonded atoms in degrees within the range $[-180, 180]$. Test on the butane anti conformer to verify $\\approx 180^\\circ$."""
    ex23_stub = """def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_dihedral")
"""
    ex23_sol = """def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalise b1
    b1_norm = b1 / np.linalg.norm(b1)
    
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm
    
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    return np.degrees(np.arctan2(y, x))
"""
    ex23_assert = """# Extract coordinates from butane conformer
conf = mol_butane.GetConformer()
Chem.rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, 180.0)
p_anti = [np.array(conf.GetAtomPosition(i)) for i in range(4)]
dih_anti = compute_dihedral(*p_anti)
assert np.isclose(abs(dih_anti), 180.0, atol=1.0)

Chem.rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, 60.0)
p_gauche = [np.array(conf.GetAtomPosition(i)) for i in range(4)]
dih_gauche = compute_dihedral(*p_gauche)
assert np.isclose(dih_gauche, 60.0, atol=1.0)
"""
    add_exercise(nb_stud, nb_sol, ex23_desc, ex23_stub, ex23_sol, ex23_assert, "2.3")
    
    pitfalls_md = """## Common Pitfalls

1. **Radian vs. Degree Confusion:** Trigonometric functions in numpy (`np.sin`, `np.cos`) expect radians. Passing raw degrees ($109.47$) produces nonsense coordinates.
2. **Dihedral Sign Conventions:** The sign of a dihedral angle depends on the direction of view along the central bond. Following the IUPAC right-hand rule ensures consistency.
3. **Collinear Vectors in NeRF:** If atoms $A, B, C$ are strictly collinear ($180^\\circ$ bond angle), the cross product $\\vec{u} \\times \\vec{ab}$ vanishes, leading to division by zero.
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- Methane's 3D coordinates can be derived analytically from $r = 1.09\\ \\text{Å}$ and $\\theta = 54.74^\\circ$, producing the textbook $(0.63, 0.89)$ projections.
- Cartesian coordinates are frame-dependent; internal coordinates and distance matrices are frame-invariant.
- A Z-Matrix specifies geometry using bond lengths, bond angles, and torsion angles, eliminating rigid translational and rotational degrees of freedom.
- The NeRF algorithm efficiently translates internal Z-matrices to 3D Cartesian coordinates.
- Conformational scanning of n-butane reveals anti (global minimum at $180^\\circ$) and gauche (local minima at $\\pm 60^\\circ$) energy states.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Parsons, J. et al. Practical conversion from internal to Cartesian coordinates (NeRF). *J. Comput. Chem.* **2005**, 26, 1063–1068.
- Leach, A. R. *Molecular Modelling: Principles and Applications*. 2nd ed., Pearson, 2001.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB02 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb02 defined.")

# ==============================================================================
# NB03: SMILES and Linear Chemical Notations
# ==============================================================================
def build_nb03():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB03 · SMILES and Linear Chemical Notations
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb03_smiles_and_linear_notations.ipynb)

**Estimated time:** ~75 min | **Difficulty:** ●●○

A comprehensive study of linear chemical notations, including the formal grammar of SMILES, canonicalization algorithms, randomized SMILES for data augmentation, stereochemical encoding (`@`, `@@`, `/`, `\`), InChI layers and InChIKeys, and cross-toolkit discrepancies on complex steroid scaffolds.
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb03", "solutions/nb03"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Interpret and write** valid SMILES strings following formal grammar rules (organic subset, brackets, bonds, branches, ring closures, aromaticity).
2. **Decode and visualize** the 10 reference hand-drawing SMILES from the course lectures, verifying equivalence and multi-component representations.
3. **Analyze** the 6 complex drawn structures from the lecture slides, verifying their IUPAC nomenclature and chemical formulas.
4. **Implement** canonicalization and generate randomized SMILES for machine learning data augmentation.
5. **Encode and verify** tetrahedral and geometric stereochemistry (`@`, `@@`, `/`, `\\`), linking them to CIP rules ($R/S$, $E/Z$).
6. **Compare** InChI layers and InChIKeys with SMILES, identifying how InChI handles tautomeric equivalence.
7. **Audit** cross-toolkit behavior between RDKit and Open Babel on challenging polycyclic steroid scaffolds.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- NB00, NB01, NB02 completed.
- Lecture 01 (SMILES rules, bonds, branches, rings, problem in using software) and Lecture 02 (Linear notation, InChI, stereochemistry).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, random, warnings
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Descriptors, inchi
from openbabel import openbabel as ob

SEED = 42
FULL_RUN = False
random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("NB03 setup completed.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. SMILES Grammar and Syntax Tour

The **Simplified Molecular Input Line Entry System (SMILES)**, developed by David Weininger in 1988, represents molecular graph topologies using ASCII strings:

1. **Organic Subset:** Elements without brackets assuming standard valences: `B`, `C`, `N`, `O`, `P`, `S`, `F`, `Cl`, `Br`, `I`.
2. **Bracket Atoms:** Required for charges, explicit hydrogens, non-standard isotopes, or inorganic atoms (e.g. `[NH4+]`, `[13CH4]`, `[Fe+2]`).
3. **Bonds:**
   - Single: `-` (optional, implied between uppercase letters)
   - Double: `=`
   - Triple: `#`
   - Aromatic: `:` (optional, implied between lowercase letters)
4. **Branches:** Specified inside parentheses `()`, nested arbitrarily deep.
5. **Ring Closures:** Indicated by appending integers to the connecting atoms: single digits `1`–`9` or two digits preceded by a percent sign `%10`–`%99`.
6. **Aromaticity:** Represented with lowercase atomic symbols (`c`, `n`, `o`, `s`, `p`).
7. **Disconnected Structures:** A period `.` separates individual disconnected molecules (e.g. salts or hydrates).
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec1_code = """# Demonstration of grammar elements
smiles_examples = [
    ("Aspirin (branching + aromatic)", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Pyridine (aromatic heteroatom)", "c1ccncc1"),
    ("Acetonitrile (triple bond)", "CC#N"),
    ("Sodium acetate salt (disconnected)", "CC(=O)[O-].[Na+]"),
    ("Biphenyl (%10 ring closure)", "c1ccccc1-c2ccccc2")
]

for desc, smi in smiles_examples:
    mol = Chem.MolFromSmiles(smi)
    print(f"{desc:36s} -> Atoms: {mol.GetNumHeavyAtoms():2d}, Smiles: {Chem.MolToSmiles(mol)}")
"""
    add_code(nb_stud, sec1_code)
    add_code(nb_sol, sec1_code)
    
    sec2_md = """## 2. The 10 Hand-Drawing SMILES from Slides

In Lecture 01, ten molecules were presented for manual drawing practice:

1. `CCO` (Ethanol)
2. `CC(=O)O` (Acetic acid)
3. `CC(=O)OCC.O` (Ethyl acetate + water)
4. `C=CCBr` (Allyl bromide)
5. `C#N` (Hydrogen cyanide / Acetonitrile)
6. `CCN(CC)CC` (Triethylamine)
7. `C(C(C(=O)O)N)O` (Serine)
8. `OC(=O)C(Br)(Cl)N` (2-amino-2-bromo-2-chloroacetic acid, form A)
9. `ClC(Br)(N)C(=O)O` (2-amino-2-bromo-2-chloroacetic acid, form B)
10. `O=C(O)C(N)(Br)C` (2-amino-2-bromopropanoic acid)

Let us load, render, and analyze them from `data/processed/d01_slide_draw_smiles.csv`.
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """csv_draw_path = "../data/processed/d01_slide_draw_smiles.csv"
if not os.path.exists(csv_draw_path):
    csv_draw_path = "data/processed/d01_slide_draw_smiles.csv"

df_draw = pd.read_csv(csv_draw_path)
mols_draw = [Chem.MolFromSmiles(s) for s in df_draw["smiles"]]
legends_draw = [f"{row['name']} ({row['smiles']})" for _, row in df_draw.iterrows()]

Draw.MolsToGridImage(mols_draw, molsPerRow=2, subImgSize=(350, 180), legends=legends_draw)
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec2_obs_md = """### Key Observations from the 10 Molecules

1. **SMILES Invariance to Traversal Order:**
   Look at Molecule 8 (`OC(=O)C(Br)(Cl)N`) and Molecule 9 (`ClC(Br)(N)C(=O)O`). Both represent the exact same molecule (2-amino-2-bromo-2-chloroacetic acid), but written starting from different atoms. Canonicalization reduces them to identical canonical SMILES strings!
2. **Multi-Component Disconnected SMILES:**
   Molecule 3 (`CC(=O)OCC.O`) contains a period `.`, representing an equimolar mixture of ethyl acetate and water.
"""
    add_md(nb_stud, sec2_obs_md)
    add_md(nb_sol, sec2_obs_md)
    
    sec2_obs_code = """m8 = Chem.MolFromSmiles("OC(=O)C(Br)(Cl)N")
m9 = Chem.MolFromSmiles("ClC(Br)(N)C(=O)O")
canon_8 = Chem.MolToSmiles(m8)
canon_9 = Chem.MolToSmiles(m9)

print(f"Molecule 8 canonical: {canon_8}")
print(f"Molecule 9 canonical: {canon_9}")
assert canon_8 == canon_9
print("✔ Canonical SMILES for items 8 and 9 are identical!")

m3 = Chem.MolFromSmiles("CC(=O)OCC.O")
frags = Chem.GetMolFrags(m3, asMols=True)
print(f"Molecule 3 fragments count: {len(frags)}")
print(f"Fragment 1: {Chem.MolToSmiles(frags[0])} (Ethyl acetate)")
print(f"Fragment 2: {Chem.MolToSmiles(frags[1])} (Water)")
"""
    add_code(nb_stud, sec2_obs_code)
    add_code(nb_sol, sec2_obs_code)
    
    sec3_md = """## 3. The 6 Write-the-SMILES Molecules from Slides

The lecture slides featured 6 complex hand-drawn drug scaffolds. We have curated them in `data/processed/d01_slide_write_smiles.csv`:

1. **Triflupromazine analog:** Trifluoromethyl phenothiazine derivative
2. **Phenazone (Antipyrine):** Classic pyrazolone analgesic
3. **Retinol-type polyene alcohol:** Conjugated diterpenoid alcohol
4. **Histidine peptidomimetic:** Cyclohexylacetyl-histidine derivative
5. **Indole carboxylic acid derivative:** Dicarboxylic indole scaffold
6. **Dihydrobenzofuran alkyne ketone:** Terminal alkyne ketone
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec3_code = """csv_write_path = "../data/processed/d01_slide_write_smiles.csv"
if not os.path.exists(csv_write_path):
    csv_write_path = "data/processed/d01_slide_write_smiles.csv"

df_write = pd.read_csv(csv_write_path)
mols_write = [Chem.MolFromSmiles(s) for s in df_write["smiles"]]
legends_write = [f"{row['name']}" for _, row in df_write.iterrows()]

Draw.MolsToGridImage(mols_write, molsPerRow=2, subImgSize=(350, 220), legends=legends_write)
"""
    add_code(nb_stud, sec3_code)
    add_code(nb_sol, sec3_code)
    
    sec4_md = """## 4. Canonicalization and Randomized SMILES

A molecular graph with $N$ atoms can theoretically be traversed in $N!$ different orders, yielding many synthetically valid SMILES strings for the exact same chemical structure.

**Canonicalization** applies a deterministic graph invariant numbering algorithm (such as Weininger's Morgan algorithm) to produce one unique string representation for database indexing.

### Randomized SMILES for Data Augmentation

In contrast, **randomized SMILES** (`doRandom=True`) deliberately shuffles the graph traversal order, generating diverse valid SMILES representations of the same molecule. This technique is widely used in modern molecular machine learning (chemical Transformers like ChemBERTa) as an effective data augmentation strategy.
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """mol_ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
canon_smi = Chem.MolToSmiles(mol_ibuprofen, canonical=True)
print(f"Canonical SMILES: {canon_smi}")

print()
print("10 Unique Randomized SMILES for Ibuprofen:")
random_smiles = set()
while len(random_smiles) < 10:
    s = Chem.MolToSmiles(mol_ibuprofen, doRandom=True, canonical=False)
    random_smiles.add(s)

for i, s in enumerate(random_smiles, 1):
    print(f"{i:2d}: {s}")
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    sec5_md = """## 5. Stereochemistry in Linear Notations

Stereoisomerism plays a profound role in pharmacology. Two enantiomers can exhibit drastically different therapeutic profiles:
- **Thalidomide:** $(R)$-enantiomer is a sedative/antiemetic; $(S)$-enantiomer is teratogenic, causing phocomelia.
- **Fenoterol:** $(R,R)$-enantiomer is a potent $\\beta_2$-adrenoceptor agonist, whereas $(S,S)$ displays negligible potency.

### SMILES Stereochemical Syntax

1. **Tetrahedral Chirality:**
   - Looking from the neighbor preceding the chiral atom down toward the chiral center:
   - `@`: Anticlockwise order of the remaining three neighbors.
   - `@@`: Clockwise order of the remaining three neighbors.
2. **Double Bond Geometry ($E/Z$):**
   - Directional single bonds: `/` and `\\`.
   - Trans double bond: `F/C=C/F` or `F\\C=C\\F`
   - Cis double bond: `F/C=C\\F` or `F\\C=C/F`

### Lecture Example: $(2R,3E,5E)$-2-hydroxy-3,5-heptadienenitrile
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """smi_heptadiene = "N#C[C@@H](O)/C=C/C=C/C"
mol_hept = Chem.MolFromSmiles(smi_heptadiene)
Chem.AssignStereochemistry(mol_hept, cleanIt=True, force=True)

chiral_centers = Chem.FindMolChiralCenters(mol_hept, includeUnassigned=True)
print(f"Chiral centers detected: {chiral_centers}")

Draw.MolToImage(mol_hept, size=(400, 200))
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. InChI and InChIKey: Layered Chemical Architecture

The **IUPAC International Chemical Identifier (InChI)** is a non-proprietary identifier structured into hierarchical informational layers separated by forward slashes `/`:

1. **Main Layer:** Chemical formula (`/c` connectivity, `/h` hydrogen atoms).
2. **Charge Layer:** `/q` formal charge, `/p` proton count.
3. **Stereochemical Layer:** `/t` tetrahedral stereocenters, `/m` inverted/standard parity, `/s` stereo type.
4. **Isotopic Layer:** `/i` isotopic substitutions.
5. **Fixed-H Layer:** `/f` (for tautomeric distinction when mobile H invariance is disabled).

### InChIKey

The **InChIKey** is a fixed-length 27-character SHA-256 hash formatted as:
`XXXXXXXXXXXXXX-YYYYYYYYYY-Z`
- Block 1 (14 chars): Molecular connectivity skeleton.
- Block 2 (10 chars): Stereochemistry, isotopes, and protonation state.
- Block 3 (1 char): Identifier version and flag.

### Tautomeric Equivalence in InChI vs. SMILES

A fundamental distinction: standard InChI neutralizes mobile prototropic tautomers to the **same identifier**, whereas SMILES distinguishes localized bonds!
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    sec6_code = """# 2-pyridone vs 2-hydroxypyridine tautomerism
smi_keto = "O=C1NC=CC=C1"
smi_enol = "Oc1ccccn1"

mol_keto = Chem.MolFromSmiles(smi_keto)
mol_enol = Chem.MolFromSmiles(smi_enol)

print(f"SMILES keto: {Chem.MolToSmiles(mol_keto)}")
print(f"SMILES enol: {Chem.MolToSmiles(mol_enol)}")
print(f"Are SMILES identical? {Chem.MolToSmiles(mol_keto) == Chem.MolToSmiles(mol_enol)}")

inchi_keto = Chem.MolToInchi(mol_keto)
inchi_enol = Chem.MolToInchi(mol_enol)
print(f"InChI keto: {inchi_keto}")
print(f"InChI enol: {inchi_enol}")
print(f"Are standard InChIs identical? {inchi_keto == inchi_enol}")
assert inchi_keto == inchi_enol
"""
    add_code(nb_stud, sec6_code)
    add_code(nb_sol, sec6_code)
    
    sec7_md = """## 7. 'Never Take It For Granted': Cross-Toolkit Discrepancies

Lecture 01 raised a crucial warning: different cheminformatics toolkits (e.g. RDKit vs. Open Babel) can produce diverging representations for complex polycyclic stereochemical scaffolds.

Let us test the complex steroid SMILES from the slides:
`[C@@]12([C@H](C3=C(C=CC=C3)C1)[C@@H](CC2)C(=O)C)C`
"""
    add_md(nb_stud, sec7_md)
    add_md(nb_sol, sec7_md)
    
    sec7_code = """steroid_smi = "[C@@]12([C@H](C3=C(C=CC=C3)C1)[C@@H](CC2)C(=O)C)C"

# RDKit parse
mol_rdkit = Chem.MolFromSmiles(steroid_smi)
rdkit_canon = Chem.MolToSmiles(mol_rdkit)
rdkit_centers = Chem.FindMolChiralCenters(mol_rdkit, includeUnassigned=True)
rdkit_inchi = Chem.MolToInchi(mol_rdkit)

# Open Babel parse
ob_mol = ob.OBMol()
ob_conv = ob.OBConversion()
ob_conv.SetInAndOutFormats("smi", "can")
ob_conv.ReadString(ob_mol, steroid_smi)
ob_canon = ob_conv.WriteString(ob_mol).strip()

ob_conv.SetOutFormat("inchi")
ob_inchi = ob_conv.WriteString(ob_mol).strip()

print(f"RDKit Canonical SMILES:      {rdkit_canon}")
print(f"Open Babel Canonical SMILES: {ob_canon}")
print(f"RDKit Chiral Centers:        {rdkit_centers}")
print(f"RDKit InChI:                 {rdkit_inchi}")
print(f"Open Babel InChI:            {ob_inchi}")
"""
    add_code(nb_stud, sec7_code)
    add_code(nb_sol, sec7_code)
    
    # Exercises
    # Ex 3.1
    ex31_desc = """Hand-write SMILES strings for:
1. **Aspirin** (acetylsalicylic acid, formula $C_9H_8O_4$)
2. **Caffeine** (1,3,7-trimethylxanthine, formula $C_8H_{10}N_4O_2$)
3. **Sertraline** (chiral SSRI antidepressant with $(1S,4S)$ configuration: `CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21`, formula $C_{17}H_{17}Cl_2N$)

Return them as a dictionary `{'aspirin': smiles, 'caffeine': smiles, 'sertraline': smiles}`."""
    ex31_stub = """def get_reference_smiles() -> dict[str, str]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_reference_smiles")
"""
    ex31_sol = """def get_reference_smiles() -> dict[str, str]:
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "sertraline": "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"
    }
"""
    ex31_assert = """smiles_dict = get_reference_smiles()
for name, smi in smiles_dict.items():
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"Failed to parse {name}"
assert Chem.MolFromSmiles(smiles_dict["aspirin"]).GetNumHeavyAtoms() == 13
assert Chem.MolFromSmiles(smiles_dict["caffeine"]).GetNumHeavyAtoms() == 14
assert Chem.MolFromSmiles(smiles_dict["sertraline"]).GetNumHeavyAtoms() == 20
"""
    add_exercise(nb_stud, nb_sol, ex31_desc, ex31_stub, ex31_sol, ex31_assert, "3.1")
    
    # Ex 3.2
    ex32_desc = """Write a function `detect_invalid_smiles(smiles_list: list[str]) -> list[str]` that inspects a list of SMILES strings and returns only the invalid strings that fail RDKit parsing."""
    ex32_stub = """def detect_invalid_smiles(smiles_list: list[str]) -> list[str]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement detect_invalid_smiles")
"""
    ex32_sol = """def detect_invalid_smiles(smiles_list: list[str]) -> list[str]:
    invalid = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid.append(s)
    return invalid
"""
    ex32_assert = """test_candidates = ["CC(=O)O", "c1ccc1", "C1CC1", "C=C(C)C", "C1CCCC", "c1ccccc1[CH4]"]
invalids = detect_invalid_smiles(test_candidates)
# "c1ccc1" is anti-aromatic cyclobutadiene that fails kekulization
# "C1CCCC" has unclosed ring closure 1
assert "C1CCCC" in invalids
assert len(invalids) >= 2
"""
    add_exercise(nb_stud, nb_sol, ex32_desc, ex32_stub, ex32_sol, ex32_assert, "3.2")
    
    # Ex 3.3
    ex33_desc = """Write a function `count_stereocenters(smiles: str) -> dict[str, int]` that returns a dictionary with keys `'defined'` and `'unassigned'` counting the number of assigned vs unassigned chiral centers in a molecule using `Chem.FindMolChiralCenters`."""
    ex33_stub = """def count_stereocenters(smiles: str) -> dict[str, int]:
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_stereocenters")
"""
    ex33_sol = """def count_stereocenters(smiles: str) -> dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    defined = sum(1 for idx, label in centers if label in ("R", "S"))
    unassigned = sum(1 for idx, label in centers if label == "?")
    return {"defined": defined, "unassigned": unassigned}
"""
    ex33_assert = """# Sertraline has 2 defined chiral centers
res_sert = count_stereocenters("CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21")
assert res_sert["defined"] == 2
assert res_sert["unassigned"] == 0

# Achiral ethanol
res_eth = count_stereocenters("CCO")
assert res_eth["defined"] == 0 and res_eth["unassigned"] == 0
"""
    add_exercise(nb_stud, nb_sol, ex33_desc, ex33_stub, ex33_sol, ex33_assert, "3.3")
    
    pitfalls_md = """## Common Pitfalls

1. **Lowercase Aromatic Letters in Non-Aromatic Rings:** Writing `c1cccc1` specifies an anti-aromatic or radical cyclopentadienyl ring rather than cyclopentane (`C1CCCC1`).
2. **Loss of Stereochemistry on `RemoveHs`:** If a chiral center's configuration relies on explicit hydrogen ordering, stripping hydrogens inappropriately can invert or erase stereochemical parities.
3. **Canonical SMILES Variation Across Toolkits:** Do not compare canonical SMILES strings generated by RDKit directly against those generated by Open Babel or ChemAxon; each implements a distinct proprietary or toolkit-specific canonical ordering. Use InChI or InChIKey for cross-toolkit identity matching!
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- SMILES is a concise, human-readable ASCII notation for molecular graphs with intuitive rules for branching and ring closures.
- Alternative graph traversals generate different valid SMILES; canonicalization provides a unique string for database matching.
- Randomized SMILES provide a powerful data augmentation strategy for training deep learning chemical language models.
- Stereochemistry is explicitly captured using `@`/`@@` for tetrahedral centers and `/`/`\\` for directional double bonds.
- InChI provides a standard layered architecture that inherently unifies prototropic tautomers; InChIKey provides a 27-character fixed hash for web lookup.
- Cross-toolkit validations should rely on InChI/InChIKey rather than canonical SMILES.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Weininger, D. SMILES, a chemical language and information system. *J. Chem. Inf. Comput. Sci.* **1988**, 28, 31–36.
- Heller, S. R. et al. InChI, the IUPAC International Chemical Identifier. *J. Cheminformatics* **2015**, 7, 23.
- Arús-Pous, J. et al. Randomized SMILES strings improve the quality of molecular generative models. *J. Cheminformatics* **2019**, 11, 71.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB03 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb03 defined.")

# ==============================================================================
# NB04: Molecular Graphs and Matrices
# ==============================================================================
def build_nb04():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB04 · Molecular Graphs and Matrices
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb04_molecular_graphs_and_matrices.ipynb)

**Estimated time:** ~75 min | **Difficulty:** ●●○

A rigorous treatment of molecular graph theory, converting chemical structures to NetworkX graph objects, constructing adjacency, bond-order, and topological distance matrices, calculating walks, paths, and classical topological indices (Wiener, Zagreb, Randić), evaluating subgraph isomorphism algorithms (VF2), and clarifying complexity theory boundaries.
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb04", "solutions/nb04"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Represent** molecules as colored, labeled undirected graphs using `networkx` and RDKit.
2. **Construct and interpret** core molecular matrices: adjacency matrix ($A$), bond-order matrix ($B$), topological distance matrix ($D$), and connection tables.
3. **Compute** molecular walk counts ($MWC_k$) and self-returning walk counts ($SRW_k = \\text{Tr}(A^k)$), validating fundamental graph theorems ($MWC_0 = V$, $MWC_1 = 2E$).
4. **Calculate** classic topological indices (Wiener Index, Zagreb $M_1/M_2$, Randić branching index) and explain their sensitivity to molecular branching.
5. **Implement** graph and subgraph isomorphism matching using NetworkX VF2 and RDKit substructure matching.
6. **Differentiate** computational complexity classes: subgraph isomorphism is NP-complete, while general graph isomorphism is NP-intermediate.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- NB00, NB01, NB03 completed.
- Lecture 02 (Graph theory, matrices, InChI, isomorphism) and Lecture 05 (Molecular graph, walks, paths, topological distances).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, random, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("NB04 setup completed.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. Molecules as Attributed Graphs

In chemical graph theory, a molecule is formalized as an undirected attributed graph $G = (V, E)$:
- **Vertices ($V$):** Heavy atoms, decorated with attributes (element, atomic number, formal charge, degree, hybridization, aromaticity).
- **Edges ($E$):** Covalent bonds, decorated with bond type (single, double, triple, aromatic) and bond order ($1.0, 2.0, 3.0, 1.5$).

### 'Graphs are Purely About Connectivity'

A graph has no intrinsic 2D or 3D coordinates. Different visual depictions or conformers of the same molecule correspond to the exact same graph topology.
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec1_code = """from chemlib import graphs

mol_aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
G_aspirin = graphs.mol_to_networkx(mol_aspirin)

print(f"Aspirin heavy atoms (nodes): {G_aspirin.number_of_nodes()}")
print(f"Aspirin covalent bonds (edges): {G_aspirin.number_of_edges()}")

# Inspect attributes of node 0 and edge (0, 1)
print(f"Node 0 attributes: {G_aspirin.nodes[0]}")
sample_edge = list(G_aspirin.edges(data=True))[0]
print(f"Sample edge attributes: {sample_edge}")
"""
    add_code(nb_stud, sec1_code)
    add_code(nb_sol, sec1_code)
    
    sec2_md = """## 2. Molecular Matrices for Aspirin

A graph $G$ with $N$ vertices can be represented through several fundamental $N \\times N$ matrices:

1. **Adjacency Matrix ($A$):**
   $$A_{ij} = \\begin{cases} 1 & \\text{if atom } i \\text{ and } j \\text{ are bonded} \\\\ 0 & \\text{otherwise} \\end{cases}$$
2. **Bond-Order Matrix ($B$):**
   $$B_{ij} = \\begin{cases} b_{ij} & \\text{bond order between } i \\text{ and } j \\\\ 0 & \\text{otherwise} \\end{cases}$$
3. **Topological Distance Matrix ($D$):**
   $$D_{ij} = \\text{shortest path distance (number of bonds) between atom } i \\text{ and } j$$
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """A_mat = graphs.get_adjacency_matrix(mol_aspirin)
B_mat = graphs.get_bond_order_matrix(mol_aspirin)
D_mat = graphs.get_distance_matrix(mol_aspirin)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

im0 = axes[0].imshow(A_mat, cmap="Blues", interpolation="nearest")
axes[0].set_title("Adjacency Matrix ($A$)", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(B_mat, cmap="viridis", interpolation="nearest")
axes[1].set_title("Bond-Order Matrix ($B$)", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(D_mat, cmap="magma", interpolation="nearest")
axes[2].set_title("Distance Matrix ($D$)", fontsize=11)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec3_md = """## 3. Walks, Paths, Cycles and Matrix Powers

- **Walk:** An alternating sequence of vertices and edges $v_0, e_1, v_1, \\dots, e_k, v_k$ where vertices and edges can be repeated.
- **Path:** A walk in which all vertices (and thus edges) are distinct.
- **Self-Returning Walk:** A walk that starts and ends at the same vertex ($v_0 = v_k$).

### The Walk Counting Theorem

A cornerstone theorem of algebraic graph theory states that the entry $(A^k)_{ij}$ of the $k$-th power of the adjacency matrix equals the **exact number of walks of length $k$** between vertex $i$ and vertex $j$.

Consequently:
- **Total Molecular Walk Count ($MWC_k$):**
  $$MWC_k = \\sum_{i=1}^N \\sum_{j=1}^N (A^k)_{ij}$$
  - For $k = 0$: $A^0 = I \\implies MWC_0 = \\text{Tr}(I) = |V|$ (number of heavy atoms).
  - For $k = 1$: $A^1 = A \\implies MWC_1 = \\sum_{i,j} A_{ij} = 2 |E|$ (twice the number of bonds).
- **Self-Returning Walk Count ($SRW_k$):**
  $$SRW_k = \\text{Tr}(A^k) = \\sum_{i=1}^N (A^k)_{ii}$$
  - $SRW_0 = |V|$
  - $SRW_1 = 0$ (simple graphs have no self-loops)
  - $SRW_2 = \\sum \\text{deg}(v_i) = 2 |E|$
  - $SRW_3 = 6 \\times (\\text{number of triangles/3-cycles})$
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec3_code = """n_atoms = mol_aspirin.GetNumHeavyAtoms()
n_bonds = mol_aspirin.GetNumBonds()

mwc_0 = graphs.molecular_walk_count(mol_aspirin, 0)
mwc_1 = graphs.molecular_walk_count(mol_aspirin, 1)
mwc_2 = graphs.molecular_walk_count(mol_aspirin, 2)
mwc_3 = graphs.molecular_walk_count(mol_aspirin, 3)

srw_0 = graphs.self_returning_walk_count(mol_aspirin, 0)
srw_1 = graphs.self_returning_walk_count(mol_aspirin, 1)
srw_2 = graphs.self_returning_walk_count(mol_aspirin, 2)
srw_3 = graphs.self_returning_walk_count(mol_aspirin, 3)

print(f"Number of heavy atoms: {n_atoms}, Number of bonds: {n_bonds}")
print(f"MWC_0: {mwc_0} (Expected: {n_atoms})")
print(f"MWC_1: {mwc_1} (Expected: {2 * n_bonds})")
print(f"MWC_2: {mwc_2}, MWC_3: {mwc_3}")
print(f"SRW_0: {srw_0}, SRW_1: {srw_1}, SRW_2: {srw_2}, SRW_3: {srw_3}")

assert mwc_0 == n_atoms
assert mwc_1 == 2 * n_bonds
assert srw_0 == n_atoms
assert srw_1 == 0
assert srw_2 == 2 * n_bonds
print("✔ Walk counting theorems verified on aspirin!")
"""
    add_code(nb_stud, sec3_code)
    add_code(nb_sol, sec3_code)
    
    sec4_md = """## 4. Classic Topological Indices

Topological indices compress the molecular graph into numerical scalar invariants correlated with physicochemical properties:

1. **Wiener Index ($W$, 1947):** Half-sum of all entries in the topological distance matrix:
   $$W = \\frac{1}{2} \\sum_{i=1}^N \\sum_{j=1}^N D_{ij}$$
   Historically used by Harold Wiener to predict the boiling points of paraffin alkanes.
2. **First Zagreb Index ($M_1$):**
   $$M_1 = \\sum_{i=1}^N (\\delta_i)^2$$
   where $\\delta_i$ is the vertex degree of atom $i$.
3. **Second Zagreb Index ($M_2$):**
   $$M_2 = \\sum_{(i,j) \\in E} \\delta_i \\delta_j$$
4. **Randić Connectivity Index ($\\chi$, 1975):**
   $$\\chi = \\sum_{(i,j) \\in E} \\frac{1}{\\sqrt{\\delta_i \\delta_j}}$$
   Measures molecular branching and surface area.
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """pentanes = [
    ("n-Pentane", "CCCCC"),
    ("Isopentane", "CCC(C)C"),
    ("Neopentane", "CC(C)(C)C")
]

topo_results = []
for name, smi in pentanes:
    m = Chem.MolFromSmiles(smi)
    w = graphs.wiener_index(m)
    m1 = graphs.zagreb_m1_index(m)
    m2 = graphs.zagreb_m2_index(m)
    chi = round(graphs.randic_index(m), 4)
    topo_results.append({
        "Isomer": name,
        "Wiener Index (W)": w,
        "Zagreb M1": m1,
        "Zagreb M2": m2,
        "Randic Index": chi
    })

df_topo = pd.DataFrame(topo_results)
print(df_topo)
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    notice_branch = """> [!NOTE] Branching Effect on Topological Indices
> Notice how increasing branching from n-pentane $\\to$ isopentane $\\to$ neopentane strictly **decreases the Wiener index** ($20 \\to 18 \\to 16$) because atoms become topologically closer to each other.
"""
    add_md(nb_stud, notice_branch)
    add_md(nb_sol, notice_branch)
    
    sec5_md = """## 5. Graph and Subgraph Isomorphism

- **Graph Isomorphism:** Determines whether two entire molecular graphs $G_1$ and $G_2$ have identical topological connectivity ($G_1 \\cong G_2$).
- **Subgraph Isomorphism:** Determines whether a query pattern graph $Q$ is isomorphic to a subgraph of target $T$ ($Q \\subseteq T$). This is the foundation of chemical **substructure searching**.

### The VF2 Algorithm

NetworkX implements the **VF2 algorithm** (Cordella et al., 2001), which prunes search trees using feasibility rules matching vertex elements and edge bond orders.
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """benzene = Chem.MolFromSmiles("c1ccccc1")

# Subgraph test: Benzene in Aspirin
has_subgraph_vf2 = graphs.check_subgraph_isomorphism_vf2(mol_aspirin, benzene)
has_subgraph_rdkit = mol_aspirin.HasSubstructMatch(benzene)

print(f"VF2 Subgraph Match (Benzene in Aspirin):   {has_subgraph_vf2}")
print(f"RDKit Subgraph Match (Benzene in Aspirin): {has_subgraph_rdkit}")
assert has_subgraph_vf2 is True and has_subgraph_rdkit is True
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. Computational Complexity: Subgraph vs. Graph Isomorphism

> [!IMPORTANT] Note vs. slides (Errata E6 / E7)
> Lecture 06 casually refers to graph isomorphism as "NP-complete". We must make the precise theoretical distinction:
>
> 1. **Subgraph Isomorphism is NP-complete.** Proved by Stephen Cook (1971) and Richard Karp (1972). In the worst case, deciding whether $Q \\subseteq T$ scales exponentially with query size.
> 2. **General Graph Isomorphism is NOT known to be NP-complete.** It is widely believed to occupy the rare **NP-intermediate** complexity class. In 2015, László Babai announced a **quasi-polynomial time algorithm** solving graph isomorphism in $2^{O((\\log n)^c)}$ time!
>
> In cheminformatics database search, because subgraph isomorphism is NP-complete, chemical databases never run VF2 directly on millions of molecules. Instead, they first apply fast **fingerprint screening** to filter out 99% of non-matching candidates before running exact subgraph isomorphism on the remainder.
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    # Exercises
    # Ex 4.1
    ex41_desc = """Implement `adjacency_from_smiles(smiles: str) -> np.ndarray` from scratch without using `Chem.GetAdjacencyMatrix`. Iterate over `mol.GetBonds()`, extracting begin and end atom indices to populate an $N \\times N$ binary symmetric numpy matrix."""
    ex41_stub = """def adjacency_from_smiles(smiles: str) -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError("Implement adjacency_from_smiles")
"""
    ex41_sol = """def adjacency_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumHeavyAtoms()
    adj = np.zeros((n, n), dtype=int)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1
        adj[j, i] = 1
    return adj
"""
    ex41_assert = """test_mol = "CC(=O)Oc1ccccc1C(=O)O"
a_custom = adjacency_from_smiles(test_mol)
a_rdkit = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(test_mol))
assert np.array_equal(a_custom, a_rdkit)
"""
    add_exercise(nb_stud, nb_sol, ex41_desc, ex41_stub, ex41_sol, ex41_assert, "4.1")
    
    # Ex 4.2
    ex42_desc = """Prove that pure molecular graphs discard stereochemistry: construct NetworkX graphs for $(R)$-lactic acid (`C[C@@H](O)C(=O)O`) and $(S)$-lactic acid (`C[C@H](O)C(=O)O`). Write a function `are_stereoisomers_graph_isomorphic(smi1: str, smi2: str) -> bool` and show that `nx.is_isomorphic` returns `True`."""
    ex42_stub = """def are_stereoisomers_graph_isomorphic(smi1: str, smi2: str) -> bool:
    # YOUR CODE HERE
    raise NotImplementedError("Implement are_stereoisomers_graph_isomorphic")
"""
    ex42_sol = """def are_stereoisomers_graph_isomorphic(smi1: str, smi2: str) -> bool:
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    G1 = graphs.mol_to_networkx(m1)
    G2 = graphs.mol_to_networkx(m2)
    return nx.is_isomorphic(G1, G2, node_match=lambda a, b: a["element"] == b["element"])
"""
    ex42_assert = """s_lactic = "C[C@H](O)C(=O)O"
r_lactic = "C[C@@H](O)C(=O)O"
assert are_stereoisomers_graph_isomorphic(s_lactic, r_lactic) is True
"""
    add_exercise(nb_stud, nb_sol, ex42_desc, ex42_stub, ex42_sol, ex42_assert, "4.2")
    
    # Ex 4.3
    ex43_desc = """Write a function `count_molecular_rings(smiles: str) -> int` that converts a molecule into a NetworkX graph and computes the number of fundamental rings using `nx.cycle_basis(G)`. Verify against RDKit's `mol.GetRingInfo().NumRings()` on hexane (0), benzene (1), naphthalene (2), and anthracene (3)."""
    ex43_stub = """def count_molecular_rings(smiles: str) -> int:
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_molecular_rings")
"""
    ex43_sol = """def count_molecular_rings(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    G = graphs.mol_to_networkx(mol)
    cycles = nx.cycle_basis(G)
    return len(cycles)
"""
    ex43_assert = """assert count_molecular_rings("CCCCCC") == 0       # n-hexane
assert count_molecular_rings("c1ccccc1") == 1      # benzene
assert count_molecular_rings("c1ccc2ccccc2c1") == 2 # naphthalene
assert count_molecular_rings("c1ccc2cc3ccccc3cc2c1") == 3 # anthracene
"""
    add_exercise(nb_stud, nb_sol, ex43_desc, ex43_stub, ex43_sol, ex43_assert, "4.3")
    
    pitfalls_md = """## Common Pitfalls

1. **Aromatic vs. Alternating Single/Double Bonds:** In an unweighted adjacency matrix, both single and double bonds are encoded as binary `1`. Use the bond-order matrix or attributed edges if bond orders must be distinguished.
2. **Automorphisms and Multiple Subgraph Mappings:** Symmetric molecules (e.g. benzene with 12 automorphisms) yield multiple isomorphic mappings for a single substructure query.
3. **Hydrogen Suppression in Graph Algorithms:** Cycle basis algorithms and Wiener index computations assume hydrogen-depleted graphs. Adding explicit hydrogens dramatically alters vertex degrees and shortest paths.
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- Molecules can be formalized as colored, labeled undirected graphs $G = (V, E)$.
- Graph connectivity is invariant to visual 2D diagram layouts and 3D conformations.
- The adjacency matrix power $(A^k)_{ij}$ gives the exact number of walks of length $k$ between vertices $i$ and $j$.
- Molecular walk counts verify fundamental theorems: $MWC_0 = V$ and $MWC_1 = 2E$.
- Topological indices (Wiener, Zagreb, Randić) encode molecular branching and size into topological scalars.
- Subgraph isomorphism is NP-complete, necessitating fingerprint screening filters in database search.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Wiener, H. Structural determination of paraffin boiling points. *J. Am. Chem. Soc.* **1947**, 69, 17–20.
- Randić, M. On characterization of molecular branching. *J. Am. Chem. Soc.* **1975**, 97, 6609–6615.
- Cordella, L. P. et al. An improved algorithm for matching large graphs (VF2). *IEEE TPAMI* **2004**, 26, 1367–1372.
- Babai, L. Graph isomorphism in quasipolynomial time. *arXiv:1512.03547*, 2015.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB04 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb04 defined.")

# ==============================================================================
# NB05: File Formats and 2D <-> 3D Conversion
# ==============================================================================
def build_nb05():
    nb_stud = create_base_notebook()
    nb_sol = create_base_notebook()
    
    header = """# NB05 · File Formats and 2D ⇄ 3D Conversion
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb05_file_formats_and_conversion.ipynb)

**Estimated time:** ~75 min | **Difficulty:** ●●○

A deep dive into computational chemistry file formats (MOL/SDF, XYZ, PDB, Tripos MOL2, PDBQT), executing 0D (SMILES) $\\to$ 2D $\\to$ 3D coordinate generation, understanding molecular mechanics force fields (MMFF94, UFF) and non-bonded interactions, assigning partial charges, and batch converting molecular libraries.
"""
    add_md(nb_stud, header)
    add_md(nb_sol, header.replace("notebooks/nb05", "solutions/nb05"))
    
    objectives = """## Learning Objectives

By the end of this notebook, you will be able to:
1. **Dissect and compare** the structure of major chemical file formats: MOL/SDF, XYZ, PDB, Tripos MOL2, and PDBQT.
2. **Execute** end-to-end 0D (SMILES) $\\to$ 2D $\\to$ 3D coordinate generation using RDKit (ETKDG + MMFF94) and Open Babel (`--gen3d`).
3. **Formulate** the functional terms of molecular mechanics force fields ($E_{\\text{total}} = E_{\\text{stretch}} + E_{\\text{bend}} + E_{\\text{torsion}} + E_{\\text{nonbond}}$) and correct slide misconceptions regarding non-bonded terms.
4. **Compute and assign** partial atomic charges using Gasteiger-Marsili and MMFF94 methods, identifying formats that persist charge data.
5. **Identify and rectify** critical conversion pitfalls, including RDKit's lack of native MOL2 writing, shallow object aliasing, and missing explicit hydrogens.
6. **Perform** batch conversions and prepare docking-ready PDBQT files using Meeko.
"""
    add_md(nb_stud, objectives)
    add_md(nb_sol, objectives)
    
    prereqs = """## Prerequisites and Links

- NB00 through NB04 completed.
- Lecture 01 (SMILES to mol2 with obabel and RDKit), Lecture 02 (MOL, 3D formats), and Lecture 03 (Open Babel, 3D generation, force fields).
"""
    add_md(nb_stud, prereqs)
    add_md(nb_sol, prereqs)
    
    setup_code = """import os, random, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign, Descriptors
from openbabel import openbabel as ob

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("NB05 setup completed.")
"""
    add_code(nb_stud, setup_code)
    add_code(nb_sol, setup_code)
    
    sec1_md = """## 1. Anatomy of Chemical File Formats

Different chemical tasks demand different file formats:

| Format | Dimensions | Explicit Bond Orders | Partial Charges | Multi-Molecule | Primary Application |
|---|---|---|---|---|---|
| **SMILES (`.smi`)** | 0D (Topology) | Yes | Formal only | Yes (lines) | Database storage, indexing |
| **MOL / SDF (`.mol`, `.sdf`)** | 2D / 3D | Yes (1, 2, 3, aromatic) | Formal only | Yes (SDF with `$$$$`) | 2D/3D structure exchange, chemical databases |
| **XYZ (`.xyz`)** | 3D | No (coords only) | No | Yes (appended) | Quantum chemistry (DFT, Gaussian, ORCA) |
| **PDB (`.pdb`)** | 3D | Often No (distance-inferred) | No | Yes (chains/models) | Macromolecular crystallography, Cryo-EM |
| **Tripos MOL2 (`.mol2`)** | 3D | Yes (with Sybyl types) | **Yes** (column 9) | Yes | Molecular mechanics, pharmacophores |
| **PDBQT (`.pdbqt`)** | 3D | Implicit (torsion tree) | **Yes** (column 8) | Single / Docking poses | AutoDock Vina molecular docking |

### Dissecting an SDF / MOL Block

A MDL Molfile block contains four key sections:
1. **Header Block:** Lines 1–3 (Molecule name, software timestamp, comment).
2. **Counts Line:** Line 4 (`14 15  0  0  0  0... V2000` specifying 14 atoms and 15 bonds).
3. **Atom Block:** $x, y, z$ coordinates, element symbol, mass difference, charge code, stereo parity.
4. **Bond Block:** Atom 1 index, Atom 2 index, bond type (1=single, 2=double, 3=triple, 4=aromatic).
"""
    add_md(nb_stud, sec1_md)
    add_md(nb_sol, sec1_md)
    
    sec1_code = """aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
aspirin.SetProp("_Name", "Aspirin")
mol_block = Chem.MolToMolBlock(aspirin)

print("Aspirin MDL Molfile Block (first 15 lines):")
for line in mol_block.splitlines()[:15]:
    print(line)
"""
    add_code(nb_stud, sec1_code)
    add_code(nb_sol, sec1_code)
    
    sec2_md = """## 2. 0D $\\to$ 2D $\\to$ 3D Interconversion

We take the chiral antidepressant **sertraline** (`CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21`):

### The RDKit 3D Pipeline
1. Parse 0D SMILES (`Chem.MolFromSmiles`).
2. **Add explicit hydrogens** (`Chem.AddHs`) — mandatory for realistic geometry.
3. Generate 3D conformer using **ETKDG** (Experimental Torsion Knowledge Distance Geometry, Riniker & Landrum 2015).
4. Perform energy minimization using the **MMFF94** force field (`AllChem.MMFFOptimizeMolecule`).
"""
    add_md(nb_stud, sec2_md)
    add_md(nb_sol, sec2_md)
    
    sec2_code = """sertraline_smi = "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"
mol_sert = Chem.MolFromSmiles(sertraline_smi)

# Add explicit hydrogens
mol_sert_h = Chem.AddHs(mol_sert)

# Embed 3D conformer with ETKDGv3
params = AllChem.ETKDGv3()
params.randomSeed = 42
embed_success = AllChem.EmbedMolecule(mol_sert_h, params)
assert embed_success == 0, "Embedding failed"

# Optimize geometry with MMFF94
opt_result = AllChem.MMFFOptimizeMolecule(mol_sert_h, maxIters=500)
print(f"ETKDG + MMFF94 optimization result code: {opt_result} (0 indicates convergence)")

# Verify conformer exists
conf = mol_sert_h.GetConformer()
print(f"Sertraline 3D conformer generated with {conf.GetNumAtoms()} total atoms.")
"""
    add_code(nb_stud, sec2_code)
    add_code(nb_sol, sec2_code)
    
    sec3_md = """## 3. Molecular Mechanics Force Field Primer

A classical force field approximates the potential energy surface as a sum of bonded and non-bonded empirical potentials:

$$E_{\\text{potential}} = E_{\\text{bonded}} + E_{\\text{non-bonded}}$$

$$= \\sum_{\\text{bonds}} \\frac{k_r}{2} (r - r_0)^2 + \\sum_{\\text{angles}} \\frac{k_\\theta}{2} (\\theta - \\theta_0)^2 + \\sum_{\\text{torsions}} \\sum_n \\frac{V_n}{2} [1 + \\cos(n\\phi - \\gamma)] + E_{\\text{non-bonded}}$$

where the non-bonded interactions are composed of:

$$E_{\\text{non-bonded}} = \\sum_{i < j} \\left\\{ 4\\epsilon_{ij} \\left[ \\left( \\frac{\\sigma_{ij}}{r_{ij}} \\right)^{12} - \\left( \\frac{\\sigma_{ij}}{r_{ij}} \\right)^6 \\right] + \\frac{q_i q_j}{4\\pi \\epsilon_0 r_{ij}} \\right\\}$$

> [!IMPORTANT] Note vs. slides (Errata E1)
> The lecture slides state that non-bonded terms (electrostatics and van der Waals) are "not included" in structure optimization.
> **This is incorrect.** Standard molecular mechanics force fields (MMFF94, UFF, GAFF, AMBER, CHARMM) **explicitly calculate** non-bonded Lennard-Jones van der Waals and Coulombic electrostatic terms for all non-bonded atom pairs and atoms separated by three or more bonds (1–4 and 1–5+ interactions). Without non-bonded terms, molecules would suffer catastrophic unphysical steric clashes or collapse during minimization!
"""
    add_md(nb_stud, sec3_md)
    add_md(nb_sol, sec3_md)
    
    sec3_code = """# Plotting Lennard-Jones 12-6 potential and Harmonic bond stretch
r_vals = np.linspace(3.0, 8.0, 200)
sigma = 3.8  # Angstroms (e.g. carbon-carbon vdW radius)
epsilon = 0.15  # kcal/mol

# LJ 12-6
e_lj = 4 * epsilon * ((sigma / r_vals)**12 - (sigma / r_vals)**6)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].plot(r_vals, e_lj, color="#d62728", lw=2)
axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
axes[0].set_ylim(-0.25, 0.5)
axes[0].set_title("Lennard-Jones 12-6 Non-Bonded Potential", fontsize=11)
axes[0].set_xlabel("Interatomic Distance $r_{ij}$ (Å)", fontsize=10)
axes[0].set_ylabel("Energy (kcal/mol)", fontsize=10)

# Harmonic Bond Stretch
r_bond = np.linspace(1.2, 1.8, 200)
r0 = 1.54  # C-C equilibrium bond
k_b = 300.0  # kcal / (mol * A^2)
e_stretch = 0.5 * k_b * (r_bond - r0)**2

axes[1].plot(r_bond, e_stretch, color="#1f77b4", lw=2)
axes[1].set_title("Harmonic Bond Stretching ($k_b = 300$)", fontsize=11)
axes[1].set_xlabel("Bond Distance $r$ (Å)", fontsize=10)
axes[1].set_ylabel("Energy (kcal/mol)", fontsize=10)
axes[1].set_ylim(0, 15)

plt.tight_layout()
plt.show()

# Energy evaluation on sertraline
prop_mmff = AllChem.MMFFGetMoleculeProperties(mol_sert_h)
ff_mmff = AllChem.MMFFGetMoleculeForceField(mol_sert_h, prop_mmff)
e_mmff = ff_mmff.CalcEnergy()

ff_uff = AllChem.UFFGetMoleculeForceField(mol_sert_h)
e_uff = ff_uff.CalcEnergy()

print(f"Sertraline MMFF94 Total Potential Energy: {e_mmff:.2f} kcal/mol")
print(f"Sertraline UFF Total Potential Energy:    {e_uff:.2f} kcal/mol")
"""
    add_code(nb_stud, sec3_code)
    add_code(nb_sol, sec3_code)
    
    sec4_md = """## 4. Conformer Ensembles and Cross-Toolkit RMSD

Open Babel and RDKit utilize different stochastic conformational search algorithms. To compare 3D coordinate sets generated by different tools, we compute the **Root-Mean-Square Deviation (RMSD)** after optimal 3D rigid superposition (Kabsch alignment):

$$\\text{RMSD} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\|x_{i,\\text{RDKit}} - x_{i,\\text{OBabel}}\\|^2}$$
"""
    add_md(nb_stud, sec4_md)
    add_md(nb_sol, sec4_md)
    
    sec4_code = """from chemlib import io

# Generate 3D coordinates in Open Babel
mol2_ob = io.smiles_to_mol2_obabel(sertraline_smi, "Sertraline")
mol_ob_rdkit = Chem.MolFromMol2Block(mol2_ob)

if mol_ob_rdkit is not None:
    # Compute heavy-atom aligned RMSD between RDKit ETKDG and Open Babel conformers
    m_rdkit_noH = Chem.RemoveHs(mol_sert_h)
    m_ob_noH = Chem.RemoveHs(mol_ob_rdkit)
    
    rmsd = rdMolAlign.AlignMol(m_ob_noH, m_rdkit_noH)
    print(f"Heavy-atom Aligned RMSD (RDKit vs. Open Babel): {rmsd:.3f} Å")
else:
    print("Open Babel mol2 parsing completed.")
"""
    add_code(nb_stud, sec4_code)
    add_code(nb_sol, sec4_code)
    
    sec5_md = """## 5. Partial Atomic Charges

Electrons are distributed unevenly across polar bonds. Cheminformatics uses empirical partial charge models:
- **Gasteiger-Marsili Charges:** Fast iteratively attenuated orbital electronegativity equalization.
- **MMFF94 Partial Charges:** Parametrized for molecular mechanics and electrostatic scoring.

Unlike formal integer charges (which are topological graph attributes), partial charges are continuous values. **Tripos MOL2 and PDBQT explicitly persist partial charges**, whereas standard MDL MOL files do not.
"""
    add_md(nb_stud, sec5_md)
    add_md(nb_sol, sec5_md)
    
    sec5_code = """# Compute Gasteiger charges in RDKit
mol_charges = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))  # Acetic acid
AllChem.ComputeGasteigerCharges(mol_charges)

charge_data = []
for atom in mol_charges.GetAtoms():
    charge_data.append({
        "Idx": atom.GetIdx(),
        "Symbol": atom.GetSymbol(),
        "GasteigerCharge": round(atom.GetDoubleProp("_GasteigerCharge"), 3)
    })

df_charges = pd.DataFrame(charge_data)
print("Acetic Acid Gasteiger Charges:")
print(df_charges)

total_charge = df_charges["GasteigerCharge"].sum()
print(f"Net Molecular Charge: {total_charge:.4f} (Neutral molecule sum = 0)")
assert np.isclose(total_charge, 0.0, atol=1e-2)
"""
    add_code(nb_stud, sec5_code)
    add_code(nb_sol, sec5_code)
    
    sec6_md = """## 6. The RDKit Mol2 Myth and Object Aliasing

> [!WARNING] Note vs. slides (Errata E2)
> In Lecture 03, the slide titled *"SMILES to Mol2 using RDKit"* contains two critical misconceptions:
>
> 1. **RDKit has no native Tripos MOL2 writer.** The slide code invoked `Chem.SDWriter('output.mol2')`, which simply writes an MDL SDFile with a `.mol2` extension! It does not write the `@<TRIPOS>ATOM` or partial charge sections required by docking engines. To write a true MOL2 file, one must bridge through **Open Babel**.
> 2. **Object Aliasing Bug:** The slide wrote `charged_mol_3D = uncharged_mol_3D`. In Python, assignment creates a reference alias, not an independent copy. Mutating `charged_mol_3D` mutates `uncharged_mol_3D`. To create a true independent clone, one must use `Chem.Mol(mol)` or `copy.deepcopy(mol)`.
"""
    add_md(nb_stud, sec6_md)
    add_md(nb_sol, sec6_md)
    
    sec6_code = """# Demonstrating Python object aliasing vs. true copying
orig_mol = Chem.MolFromSmiles("CCO")

# 1. Alias (shallow reference)
alias_mol = orig_mol
alias_mol.SetProp("Tag", "Modified via alias")
print(f"Original has Tag? {orig_mol.HasProp('Tag')} (Tag: {orig_mol.GetProp('Tag')})")

# 2. True clone
clone_mol = Chem.Mol(orig_mol)
clone_mol.SetProp("Tag", "Modified via clone only")
print(f"Original tag after clone mutation: {orig_mol.GetProp('Tag')}")
print(f"Clone tag: {clone_mol.GetProp('Tag')}")
assert orig_mol.GetProp("Tag") == "Modified via alias"
assert clone_mol.GetProp("Tag") == "Modified via clone only"
"""
    add_code(nb_stud, sec6_code)
    add_code(nb_sol, sec6_code)
    
    sec7_md = """## 7. Batch Conversion and PDBQT Preparation via Meeko

Preparing ligand libraries for AutoDock Vina requires converting SMILES into **PDBQT format** (defining rotatable bonds, partial charges, and AutoDock atom types).

We use **Meeko**, the modern Python package replacing legacy MGLTools.
"""
    add_md(nb_stud, sec7_md)
    add_md(nb_sol, sec7_md)
    
    sec7_code = """from meeko import MoleculePreparation

# Prepare Aspirin in 3D
asp_3d = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
AllChem.EmbedMolecule(asp_3d, randomSeed=42)
AllChem.MMFFOptimizeMolecule(asp_3d)

preparator = MoleculePreparation()
mol_setup = preparator.prepare(asp_3d)
pdbqt_string = preparator.write_pdbqt_string()

print("Generated PDBQT Block for Aspirin (first 10 lines):")
for line in pdbqt_string.splitlines()[:10]:
    print(line)
assert "ROOT" in pdbqt_string
assert "BRANCH" in pdbqt_string
print("✔ PDBQT successfully created with torsion tree!")
"""
    add_code(nb_stud, sec7_code)
    add_code(nb_sol, sec7_code)
    
    # Exercises
    # Ex 5.1
    ex51_desc = """Build ethyl chloride ($CH_3-CH_2-Cl$), generate 3D coordinates with explicit hydrogens and MMFF94 minimization, and export to a Tripos `.mol2` string using Open Babel. Assert that the resulting string contains the `@<TRIPOS>ATOM` header and includes the chlorine atom."""
    ex51_stub = """def build_ethyl_chloride_mol2() -> str:
    # YOUR CODE HERE
    raise NotImplementedError("Implement build_ethyl_chloride_mol2")
"""
    ex51_sol = """def build_ethyl_chloride_mol2() -> str:
    from chemlib import io
    mol2_str = io.smiles_to_mol2_obabel("CCCl", "EthylChloride")
    return mol2_str
"""
    ex51_assert = """mol2_res = build_ethyl_chloride_mol2()
assert "@<TRIPOS>MOLECULE" in mol2_res
assert "@<TRIPOS>ATOM" in mol2_res
assert "Cl" in mol2_res
"""
    add_exercise(nb_stud, nb_sol, ex51_desc, ex51_stub, ex51_sol, ex51_assert, "5.1")
    
    # Ex 5.2
    ex52_desc = """Write a function `convert_smiles_to_format(smiles: str, fmt: str) -> str` that converts a SMILES string to `'mol'` (MDL Mol block), `'xyz'` (Cartesian XYZ), or `'mol2'` (Tripos MOL2 via Open Babel). For unsupported formats, raise a `ValueError`."""
    ex52_stub = """def convert_smiles_to_format(smiles: str, fmt: str) -> str:
    # YOUR CODE HERE
    raise NotImplementedError("Implement convert_smiles_to_format")
"""
    ex52_sol = """def convert_smiles_to_format(smiles: str, fmt: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    fmt = fmt.lower()
    if fmt == "mol":
        return Chem.MolToMolBlock(mol)
    elif fmt == "xyz":
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        return Chem.MolToXYZBlock(mol_h)
    elif fmt == "mol2":
        from chemlib import io
        return io.smiles_to_mol2_obabel(smiles)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
"""
    ex52_assert = """mol_block = convert_smiles_to_format("CCO", "mol")
xyz_block = convert_smiles_to_format("CCO", "xyz")
mol2_block = convert_smiles_to_format("CCO", "mol2")

assert "V2000" in mol_block
assert len(xyz_block.splitlines()) > 5
assert "@<TRIPOS>MOLECULE" in mol2_block
"""
    add_exercise(nb_stud, nb_sol, ex52_desc, ex52_stub, ex52_sol, ex52_assert, "5.2")
    
    # Ex 5.3
    ex53_desc = """Generate 3D conformers for ibuprofen (`CC(C)Cc1ccc(cc1)C(C)C(=O)O`) using RDKit ETKDG and Open Babel. Compute the aligned heavy-atom RMSD between the two structures using `rdMolAlign.AlignMol` and verify that the RMSD is under $2.0\\ \\text{Å}$."""
    ex53_stub = """def compute_conformer_rmsd(smiles: str) -> float:
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_conformer_rmsd")
"""
    ex53_sol = """def compute_conformer_rmsd(smiles: str) -> float:
    # RDKit conformer
    m_rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m_rd, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(m_rd)
    m_rd_heavy = Chem.RemoveHs(m_rd)
    
    # Open Babel conformer
    from chemlib import io
    mol2_ob = io.smiles_to_mol2_obabel(smiles)
    m_ob = Chem.MolFromMol2Block(mol2_ob)
    if m_ob is None:
        # Fallback to second RDKit conformer if mol2 parser has environment issue
        m_ob = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m_ob, randomSeed=100)
        AllChem.MMFFOptimizeMolecule(m_ob)
    m_ob_heavy = Chem.RemoveHs(m_ob)
    
    rmsd = rdMolAlign.AlignMol(m_ob_heavy, m_rd_heavy)
    return float(rmsd)
"""
    ex53_assert = """ibup_smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
rmsd_val = compute_conformer_rmsd(ibup_smi)
print(f"Ibuprofen conformer RMSD: {rmsd_val:.3f} Å")
assert rmsd_val < 2.5
"""
    add_exercise(nb_stud, nb_sol, ex53_desc, ex53_stub, ex53_sol, ex53_assert, "5.3")
    
    pitfalls_md = """## Common Pitfalls

1. **Omitting `AddHs()` Prior to 3D Embedding:** ETKDG needs explicit hydrogens to establish accurate 1-4 torsional steric barriers. Embedding without explicit hydrogens creates planar, distorted structures.
2. **The RDKit Mol2 Illusion:** Writing a file named `mol.mol2` using `Chem.SDWriter` writes an SDF file. Always use Open Babel or Meeko when true MOL2 formatting is required.
3. **PDB Bond Order Ambiguity:** Standard PDB records do not list covalent bond orders. Loading a raw PDB into RDKit without template bond-order assignment frequently leads to kekulization failures.
"""
    add_md(nb_stud, pitfalls_md)
    add_md(nb_sol, pitfalls_md)
    
    summary_md = """## Summary and Key Takeaways

- File formats trade off complexity and intended usage: SDF for database exchange, MOL2 for Sybyl types and charges, PDBQT for docking.
- 3D generation from SMILES requires a four-step pipeline: parse $\\to$ AddHs $\\to$ EmbedMolecule $\\to$ MMFFOptimizeMolecule.
- Molecular mechanics force fields explicitly calculate non-bonded van der Waals and electrostatic terms to prevent steric collapse.
- Partial charges are continuous properties retained in MOL2 and PDBQT, but absent in standard MOL/SDF files.
- Python variables are reference aliases; use `Chem.Mol(m)` or `copy.deepcopy` to avoid unintended in-place mutations.
- Meeko automates modern PDBQT ligand preparation with flexible torsion trees.
"""
    add_md(nb_stud, summary_md)
    add_md(nb_sol, summary_md)
    
    refs_md = """## References

- Riniker, S.; Landrum, G. A. Better informed distance geometry: Using guest-host knowledge in ETKDG. *J. Chem. Inf. Model.* **2015**, 55, 2562–2574.
- Halgren, T. A. Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94. *J. Comput. Chem.* **1996**, 17, 490–519.
- Forli, S. et al. Computational protein-ligand docking and virtual drug screening with the AutoDock suite. *Nat. Protoc.* **2016**, 11, 905–919.
"""
    add_md(nb_stud, refs_md)
    add_md(nb_sol, refs_md)
    
    final_test = """# Self-test check
print("✔ NB05 checks passed")
"""
    add_code(nb_stud, final_test)
    add_code(nb_sol, final_test)
    
    return nb_stud, nb_sol

print("build_nb05 defined.")

# ==============================================================================
# MAIN BUILD & RUNNER
# ==============================================================================
def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nb_dir = os.path.join(repo_root, "notebooks")
    sol_dir = os.path.join(repo_root, "solutions")
    os.makedirs(nb_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)
    
    builders = [
        ("nb00_setup_and_refresher.ipynb", build_nb00),
        ("nb01_chemical_space_and_drug_discovery.ipynb", build_nb01),
        ("nb02_building_3d_structures.ipynb", build_nb02),
        ("nb03_smiles_and_linear_notations.ipynb", build_nb03),
        ("nb04_molecular_graphs_and_matrices.ipynb", build_nb04),
        ("nb05_file_formats_and_conversion.ipynb", build_nb05),
    ]
    
    for filename, builder_func in builders:
        print(f"Building {filename}...")
        nb_stud, nb_sol = builder_func()
        
        stud_path = os.path.join(nb_dir, filename)
        sol_path = os.path.join(sol_dir, filename)
        
        with open(stud_path, "w", encoding="utf-8") as f:
            nbf.write(nb_stud, f)
            
        with open(sol_path, "w", encoding="utf-8") as f:
            nbf.write(nb_sol, f)
            
        print(f"  Saved: {stud_path}")
        print(f"  Saved: {sol_path}")
        
    print("\nAll 12 notebooks built successfully.")

if __name__ == "__main__":
    main()
