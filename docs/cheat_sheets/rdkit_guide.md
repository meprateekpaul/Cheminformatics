# RDKit Python Quick Reference

## Core Operations

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdFingerprintGenerator, DataStructs

# 1. Parsing and Writing
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
canon_smi = Chem.CanonSmiles("Oc1ccccc1C(=O)O")

# 2. Hydrogens
mol_with_h = Chem.AddHs(mol)
mol_no_h = Chem.RemoveHs(mol_with_h)

# 3. 3D Conformation Generation (ETKDG)
params = AllChem.ETKDGv3()
params.randomSeed = 42
AllChem.EmbedMolecule(mol_with_h, params)
AllChem.MMFFOptimizeMolecule(mol_with_h)

# 4. Fingerprints (Modern API)
fpgen_ecfp4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = fpgen_ecfp4.GetFingerprint(mol)

# 5. Similarity
tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
dice = DataStructs.DiceSimilarity(fp1, fp2)

# 6. Substructure Matching
query = Chem.MolFromSmarts("c1ccccc1")
matches = mol.GetSubstructMatches(query)

# 7. Descriptors
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
```
