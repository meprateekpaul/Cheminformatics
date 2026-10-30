"""chemlib.pharm - Pharmacophore feature extraction, 2D signature fingerprints, and 3-point pharmacophore modeling."""

from __future__ import annotations
import os
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D, SigFactory
from rdkit import RDConfig


# Load default RDKit BaseFeatures factory
_fdef_file = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_feat_factory = ChemicalFeatures.BuildFeatureFactory(_fdef_file)


def extract_features(mol: Chem.Mol, conf_id: int = -1) -> List[Dict[str, Any]]:
    """Extract 3D pharmacophore features from a molecule using RDKit ChemicalFeatures."""
    if mol is None:
        return []
    feats = _feat_factory.GetFeaturesForMol(mol, confId=conf_id)
    feature_list = []
    has_conf = mol.GetNumConformers() > 0
    for f in feats:
        pos = None
        if has_conf:
            try:
                p = f.GetPos()
                pos = (p.x, p.y, p.z)
            except Exception:
                pos = None
        feature_list.append({
            "family": f.GetFamily(),
            "type": f.GetType(),
            "atom_ids": list(f.GetAtomIds()),
            "pos": pos,
        })
    return feature_list


def get_pharm2d_fingerprint(mol: Chem.Mol) -> Any:
    """Generate 2D pharmacophore fingerprint (Gobbi signature) for a molecule."""
    if mol is None:
        return None
    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


def create_custom_3point_sigfactory(
    bins: Optional[List[Tuple[int, int]]] = None,
    min_point_count: int = 2,
    max_point_count: int = 3,
) -> SigFactory.SigFactory:
    """Create a customized Pharm2D SigFactory with specified distance bins."""
    if bins is None:
        # 7 standard bins: (1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)
        bins = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]
    
    sig_factory = SigFactory.SigFactory(_feat_factory, minPointCount=min_point_count, maxPointCount=max_point_count)
    sig_factory.SetBins(bins)
    sig_factory.Init()
    return sig_factory


def match_3point_pharmacophore_distances(
    mol: Chem.Mol,
    feat_types: Tuple[str, str, str],
    target_distances: Tuple[float, float, float],
    tolerance: float = 1.5,
    conf_id: int = 0,
) -> bool:
    """Check if a 3D conformer matches a 3-point pharmacophore query with distance tolerances:
    feat_types: (f1, f2, f3) e.g. ('Acceptor', 'Donor', 'Aromatic')
    target_distances: (d12, d23, d13) in Angstroms
    tolerance: +/- tolerance in Angstroms
    """
    if mol is None or mol.GetNumConformers() == 0:
        return False

    conf = mol.GetConformer(conf_id)
    feats = _feat_factory.GetFeaturesForMol(mol, confId=conf_id)

    # Group features by family
    f1_list = [f for f in feats if f.GetFamily() == feat_types[0]]
    f2_list = [f for f in feats if f.GetFamily() == feat_types[1]]
    f3_list = [f for f in feats if f.GetFamily() == feat_types[2]]

    d12_target, d23_target, d13_target = target_distances

    for f1 in f1_list:
        p1 = np.array(f1.GetPos())
        for f2 in f2_list:
            if f1.GetId() == f2.GetId():
                continue
            p2 = np.array(f2.GetPos())
            d12 = np.linalg.norm(p1 - p2)
            if abs(d12 - d12_target) > tolerance:
                continue

            for f3 in f3_list:
                if f3.GetId() in (f1.GetId(), f2.GetId()):
                    continue
                p3 = np.array(f3.GetPos())
                d23 = np.linalg.norm(p2 - p3)
                d13 = np.linalg.norm(p1 - p3)

                if abs(d23 - d23_target) <= tolerance and abs(d13 - d13_target) <= tolerance:
                    return True

    return False
