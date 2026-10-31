"""chemlib.standardize - Curation, salt stripping, neutralization, tautomerization, and deduplication."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

_salt_remover = SaltRemover()
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
_uncharger = rdMolStandardize.Uncharger()


def remove_salts(mol: Chem.Mol) -> Chem.Mol:
    """Strip salts and solvents from a molecule, retaining the largest organic fragment."""
    if mol is None:
        return None
    cleaned = _salt_remover.StripMol(mol, dontRemoveEverything=True)
    frags = Chem.GetMolFrags(cleaned, asMols=True)
    if not frags:
        return cleaned
    # Pick fragment with the most heavy atoms
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return largest


def neutralize_charges(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize simple acidic and basic formal charges using MolStandardize Uncharger."""
    if mol is None:
        return None
    try:
        return _uncharger.uncharge(mol)
    except Exception:
        return mol


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Enumerate and select the canonical tautomer using RDKit MolStandardize."""
    if mol is None:
        return None
    try:
        return _tautomer_canonicalizer.Canonicalize(mol)
    except Exception:
        return mol


def standardize_mol(
    mol: Chem.Mol, strip_salts: bool = True, neutralize: bool = True, tautomerize: bool = False
) -> Chem.Mol | None:
    """Run full standardization pipeline on an RDKit Mol."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if strip_salts:
        mol = remove_salts(mol)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None

    if neutralize:
        mol = neutralize_charges(mol)

    if tautomerize:
        mol = canonicalize_tautomer(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def standardize_smiles(smiles: str, isomeric: bool = True) -> str | None:
    """Standardize a SMILES string to canonical, salt-stripped representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    std_mol = standardize_mol(mol)
    if std_mol is None:
        return None
    return Chem.MolToSmiles(std_mol, isomericSmiles=isomeric)


def curate_bioactivity_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    activity_col: str = "standard_value",
    activity_type_col: str | None = "standard_type",
    expected_activity_type: str | None = None,
    relation_col: str | None = "standard_relation",
    active_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float = 10000.0,
) -> pd.DataFrame:
    """Curate a raw bioactivity dataset:
    1. Filter for exact relations ('=') and valid activity type
    2. Remove NaN / invalid SMILES and standardize
    3. Aggregate duplicates (median activity, flag high IQR)
    4. Compute pIC50 / pChEMBL = -log10(value * 1e-9)
    5. Assign binary activity labels (1 for active <= active_threshold, 0 for inactive >= inactive_threshold)
    """
    clean_df = df.copy()

    # Filter activity type if specified
    if expected_activity_type and activity_type_col in clean_df.columns:
        clean_df = clean_df[
            clean_df[activity_type_col].str.upper() == expected_activity_type.upper()
        ]

    # Filter relation
    if relation_col in clean_df.columns:
        clean_df = clean_df[clean_df[relation_col].isin(["=", "=="])]

    # Drop missing values
    clean_df = clean_df.dropna(subset=[smiles_col, activity_col])
    clean_df[activity_col] = pd.to_numeric(clean_df[activity_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[activity_col])
    clean_df = clean_df[clean_df[activity_col] > 0]

    # Standardize SMILES
    clean_df["std_smiles"] = clean_df[smiles_col].apply(standardize_smiles)
    clean_df = clean_df.dropna(subset=["std_smiles"])

    # Aggregate duplicates
    def aggregate_group(group: pd.DataFrame) -> pd.Series:
        vals = group[activity_col].values
        med_val = float(np.median(vals))
        count = len(vals)
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25)) if count >= 4 else 0.0
        return pd.Series({activity_col: med_val, "measurement_count": count, "activity_iqr": iqr})

    agg_df = (
        clean_df.groupby("std_smiles").apply(aggregate_group, include_groups=False).reset_index()
    )

    # Calculate pIC50 / pActivity: -log10(nM * 1e-9) = 9 - log10(nM)
    agg_df["pActivity"] = 9.0 - np.log10(agg_df[activity_col])

    # Assign classification labels
    # 1: active (<= active_threshold_nm), 0: inactive (>= inactive_threshold_nm), intermediate: -1
    labels = np.full(len(agg_df), -1, dtype=int)
    labels[agg_df[activity_col] <= active_threshold_nm] = 1
    labels[agg_df[activity_col] >= inactive_threshold_nm] = 0
    agg_df["is_active"] = labels
    agg_df["activity_label"] = labels

    return agg_df
