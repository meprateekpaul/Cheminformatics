"""chemlib.descriptors - Calculation of 0D-3D molecular descriptors, Lipinski/Veber rules, and Mordred."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

# Common functional group SMARTS patterns
FUNCTIONAL_GROUPS_SMARTS = {
    "alcohol": "[OX2H]",
    "phenol": "[OX2H][c]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6,H]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[#6][CX3](=O)[NX3H0,NX3H1,NX3H2]",
    "primary_amine": "[NX3H2;!$(NC=O)]",
    "secondary_amine": "[NX3H1;!$(NC=O)]([#6])[#6]",
    "tertiary_amine": "[NX3H0;!$(NC=O)]([#6])([#6])[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    "thiol": "[SX2H]",
    "sulfide": "[#6][SX2][#6]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "aromatic_ring": "a",
}


def compute_functional_group_counts(mol: Chem.Mol) -> dict[str, int]:
    """Count the occurrences of standard functional groups using SMARTS matching."""
    if mol is None:
        return {name: 0 for name in FUNCTIONAL_GROUPS_SMARTS}
    counts = {}
    for name, smarts in FUNCTIONAL_GROUPS_SMARTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            matches = mol.GetSubstructMatches(pat)
            counts[name] = len(matches)
        else:
            counts[name] = 0
    return counts


def compute_lipinski_veber(mol: Chem.Mol) -> dict[str, float | int | bool]:
    """Compute Lipinski's Rule of 5 and Veber rules properties:
    - MW <= 500
    - MolLogP <= 5
    - NumHDonors <= 5
    - NumHAcceptors <= 10
    - NumRotatableBonds <= 10 (Veber)
    - TPSA <= 140 (Veber)
    """
    if mol is None:
        return {}

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    tpsa = float(Descriptors.TPSA(mol))

    # Lipinski violations
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    return {
        "MW": mw,
        "MolLogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "NumRotatableBonds": rotb,
        "TPSA": tpsa,
        "Lipinski_Violations": violations,
        "Passes_Lipinski": violations <= 1,
        "Passes_Veber": (rotb <= 10) and (tpsa <= 140.0),
    }


def compute_rdkit_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    """Compute all standard 2D RDKit descriptors for a list of molecules."""
    desc_names = [d[0] for d in Descriptors._descList]
    desc_funcs = [d[1] for d in Descriptors._descList]

    data = []
    for mol in mols:
        if mol is None:
            data.append([np.nan] * len(desc_funcs))
            continue
        row = []
        for func in desc_funcs:
            try:
                row.append(func(mol))
            except Exception:
                row.append(np.nan)
        data.append(row)

    df = pd.DataFrame(data, columns=desc_names)
    return df


def compute_mordred_descriptors(mols: list[Chem.Mol], ignore_3d: bool = True) -> pd.DataFrame:
    """Compute Mordred descriptors using mordredcommunity."""
    try:
        from mordred import Calculator, descriptors

        calc = Calculator(descriptors, ignore_3D=ignore_3d)
        valid_mols = [m if m is not None else Chem.MolFromSmiles("") for m in mols]
        df = calc.pandas(valid_mols)
        # Convert errors to NaN
        df = df.apply(pd.to_numeric, errors="coerce")
        return df
    except Exception:
        # Fallback to empty DataFrame with warning
        return pd.DataFrame()


def clean_descriptor_matrix(
    df: pd.DataFrame,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
) -> pd.DataFrame:
    """Clean a descriptor matrix:
    1. Impute or drop NaN columns
    2. Drop zero-variance or near-constant columns
    3. Drop highly correlated collinear columns (|r| > correlation_threshold)
    """
    clean_df = df.copy()

    # Drop columns with > 20% missing
    clean_df = clean_df.dropna(axis=1, thresh=int(len(clean_df) * 0.8))
    # Impute remaining NaNs with column median
    clean_df = clean_df.fillna(clean_df.median(numeric_only=True))

    # Variance filter
    variances = clean_df.var(numeric_only=True)
    valid_cols = variances[variances > variance_threshold].index
    clean_df = clean_df[valid_cols]

    # Correlation filter
    corr_matrix = clean_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    clean_df = clean_df.drop(columns=to_drop)

    return clean_df
