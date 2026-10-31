"""chemlib.fingerprints - Modern fingerprint generation (rdFingerprintGenerator), similarity coefficients, and LBVS."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

# Pre-instantiate standard modern fingerprint generators
_morgan_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_morgan_r3_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_rdkit_fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
_atompair_fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
_torsion_fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)


def get_fingerprint(mol: Chem.Mol, fp_type: str = "ecfp4", n_bits: int = 2048) -> Any | None:
    """Generate a molecular fingerprint using modern RDKit generators.
    Supported types: 'ecfp4' (Morgan r=2), 'ecfp6' (Morgan r=3), 'maccs', 'rdkit', 'atompair', 'torsion'
    """
    if mol is None:
        return None

    fp_type = fp_type.lower()
    if fp_type == "ecfp4":
        if n_bits == 2048:
            return _morgan_fpgen.GetFingerprint(mol)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    elif fp_type == "ecfp6":
        if n_bits == 2048:
            return _morgan_r3_fpgen.GetFingerprint(mol)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    elif fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "rdkit":
        if n_bits == 2048:
            return _rdkit_fpgen.GetFingerprint(mol)
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
        return gen.GetFingerprint(mol)
    elif fp_type == "atompair":
        if n_bits == 2048:
            return _atompair_fpgen.GetFingerprint(mol)
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        return gen.GetFingerprint(mol)
    elif fp_type == "torsion":
        if n_bits == 2048:
            return _torsion_fpgen.GetFingerprint(mol)
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        return gen.GetFingerprint(mol)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")


def fp_to_numpy(fp: Any) -> np.ndarray:
    """Convert an RDKit ExplicitBitVect to a 1D numpy array."""
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_contingency_counts(fp1: Any, fp2: Any) -> tuple[int, int, int, int]:
    """Compute the 2x2 contingency table for two binary fingerprints:
    a: bits on in both (shared 1s)
    b: bits on in fp1 only (1, 0)
    c: bits on in fp2 only (0, 1)
    d: bits off in both (shared 0s)
    """
    v1 = fp_to_numpy(fp1)
    v2 = fp_to_numpy(fp2)
    a = int(np.sum((v1 == 1) & (v2 == 1)))
    b = int(np.sum((v1 == 1) & (v2 == 0)))
    c = int(np.sum((v1 == 0) & (v2 == 1)))
    d = int(np.sum((v1 == 0) & (v2 == 0)))
    return a, b, c, d


def tanimoto_similarity(fp1: Any, fp2: Any) -> float:
    """Calculate Tanimoto (Jaccard) similarity: a / (a + b + c)."""
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def dice_similarity(fp1: Any, fp2: Any) -> float:
    """Calculate Dice similarity: 2a / (2a + b + c)."""
    return float(DataStructs.DiceSimilarity(fp1, fp2))


def tversky_similarity(fp1: Any, fp2: Any, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Calculate Tversky similarity: a / (alpha * b + beta * c + a).
    alpha=beta=1 -> Tanimoto; alpha=beta=0.5 -> Dice.
    """
    return float(DataStructs.TverskySimilarity(fp1, fp2, alpha, beta))


def compute_similarity_coefficients(fp1: Any, fp2: Any) -> dict[str, float]:
    """Calculate standard binary similarity coefficients from Lecture 06:
    Tanimoto, Dice, Cosine/Ochiai, Simple Matching, Russell-Rao, Forbes, Simpson, Kulczynski.
    """
    a, b, c, d = compute_contingency_counts(fp1, fp2)
    p = a + b + c + d

    tanimoto = a / (a + b + c) if (a + b + c) > 0 else 0.0
    dice = (2.0 * a) / (2 * a + b + c) if (2 * a + b + c) > 0 else 0.0
    cosine = a / np.sqrt((a + b) * (a + c)) if ((a + b) * (a + c)) > 0 else 0.0
    simple_matching = (a + d) / p if p > 0 else 0.0
    russell_rao = a / p if p > 0 else 0.0
    simpson = a / min(a + b, a + c) if min(a + b, a + c) > 0 else 0.0
    kulczynski = 0.5 * (a / (a + b) + a / (a + c)) if (a + b > 0 and a + c > 0) else 0.0
    forbes = (p * a) / ((a + b) * (a + c)) if ((a + b) * (a + c)) > 0 else 0.0

    return {
        "Tanimoto": float(tanimoto),
        "Dice": float(dice),
        "Cosine_Ochiai": float(cosine),
        "Simple_Matching": float(simple_matching),
        "Russell_Rao": float(russell_rao),
        "Simpson": float(simpson),
        "Kulczynski": float(kulczynski),
        "Forbes": float(forbes),
    }


def bulk_similarity_search(
    query_fp: Any,
    library_fps: list[Any],
    metric: str = "tanimoto",
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Compute similarities between a query fingerprint and a list of library fingerprints.
    Returns sorted list of (library_idx, similarity) descending.
    """
    if metric == "tanimoto":
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, library_fps)
    elif metric == "dice":
        sims = DataStructs.BulkDiceSimilarity(query_fp, library_fps)
    else:
        sims = [tanimoto_similarity(query_fp, fp) for fp in library_fps]

    results = [(idx, s) for idx, s in enumerate(sims) if s >= threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def max_similarity_to_actives(
    library_fps: list[Any],
    active_fps: list[Any],
) -> np.ndarray:
    """Compute maximum Tanimoto similarity of each library compound to any compound in active_fps."""
    max_sims = np.zeros(len(library_fps))
    for act_fp in active_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(act_fp, library_fps))
        max_sims = np.maximum(max_sims, sims)
    return max_sims
