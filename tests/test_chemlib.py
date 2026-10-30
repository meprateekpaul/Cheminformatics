"""tests/test_chemlib.py - Unit tests for the chemlib package."""

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import chemlib
from chemlib import io, standardize, graphs, descriptors, fingerprints, metrics, pharm, docking


def test_standardize_smiles():
    # Sodium acetate salt -> acetate/acetic acid
    smi = "CC(=O)[O-].[Na+]"
    std = standardize.standardize_smiles(smi)
    assert std is not None
    assert "[Na+]" not in std


def test_graphs_matrices():
    # Aspirin: CC(=O)Oc1ccccc1C(=O)O
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    A = graphs.get_adjacency_matrix(mol)
    assert A.shape == (mol.GetNumHeavyAtoms(), mol.GetNumHeavyAtoms())
    
    # MWC checks from lecture: MWC_0 = atoms, MWC_1 = 2 * bonds
    mwc0 = graphs.molecular_walk_count(mol, 0)
    mwc1 = graphs.molecular_walk_count(mol, 1)
    assert mwc0 == mol.GetNumHeavyAtoms()
    assert mwc1 == 2 * mol.GetNumBonds()

    w = graphs.wiener_index(mol)
    assert w > 0
    m1 = graphs.zagreb_m1_index(mol)
    assert m1 > 0


def test_descriptors_lipinski():
    # Aspirin
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    props = descriptors.compute_lipinski_veber(mol)
    assert props["MW"] < 500
    assert props["HBD"] <= 5
    assert props["HBA"] <= 10
    assert props["Passes_Lipinski"] is True
    assert props["Passes_Veber"] is True


def test_fingerprints_similarity():
    mol1 = Chem.MolFromSmiles("c1ccccc1O")  # Phenol
    mol2 = Chem.MolFromSmiles("c1ccccc1N")  # Aniline
    fp1 = fingerprints.get_fingerprint(mol1, "ecfp4")
    fp2 = fingerprints.get_fingerprint(mol2, "ecfp4")
    assert fp1 is not None and fp2 is not None

    tan = fingerprints.tanimoto_similarity(fp1, fp2)
    assert 0.0 < tan < 1.0

    coeffs = fingerprints.compute_similarity_coefficients(fp1, fp2)
    assert "Tanimoto" in coeffs
    assert "Dice" in coeffs
    assert "Cosine_Ochiai" in coeffs


def test_metrics_classification_regression():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    res = metrics.classification_report_dict(y_true, y_pred)
    assert res["TP"] == 2
    assert res["FP"] == 1
    assert res["FN"] == 1
    assert res["TN"] == 2

    # Regression
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 3.2, 3.9])
    reg = metrics.regression_metrics_dict(yt, yp)
    assert reg["R2"] > 0.95
    assert reg["RMSE"] < 0.2


def test_pharm_features():
    mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
    feats = pharm.extract_features(mol)
    assert len(feats) > 0
    families = [f["family"] for f in feats]
    assert "Acceptor" in families or "Donor" in families


def test_io_conversion():
    res = io.smiles_to_mol2_obabel("CCO", "ethanol")
    assert "@<TRIPOS>MOLECULE" in res
    assert "@<TRIPOS>ATOM" in res


def test_metrics_q2_bedroc_scaffold():
    # Test q2_loo
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
    q2 = metrics.q2_loo(y_true, y_pred)
    assert q2 > 0.9

    # Test BEDROC
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    bedroc = metrics.bedroc_score(labels, scores, alpha=20.0)
    assert bedroc > 0.5

    # Test scaffold split
    smiles = [
        "c1ccccc1CC(=O)O", "c1ccccc1CO", "c1ccccc1N",  # Benzene scaffold
        "c1ccncc1CC(=O)O", "c1ccncc1CO",               # Pyridine scaffold
        "CCCC", "CCCCC", "CCCCCC"                       # Alkanes (no ring scaffold)
    ]
    train_idx, test_idx = metrics.scaffold_split(smiles, train_ratio=0.6, seed=42)
    assert len(train_idx) + len(test_idx) == len(smiles)
    assert set(train_idx).isdisjoint(set(test_idx))


def test_docking_prep_and_box():
    mol = docking.prepare_ligand_3d("CCO", seed=42)
    assert mol is not None
    assert mol.GetNumConformers() > 0

    center, sizes = docking.get_box_from_ligand_mol(mol, padding=5.0)
    assert len(center) == 3
    assert len(sizes) == 3
    assert all(s > 5.0 for s in sizes)

    # Self RMSD should be 0.0
    rmsd = docking.compute_heavy_atom_rmsd(mol, mol)
    assert abs(rmsd) < 1e-5

