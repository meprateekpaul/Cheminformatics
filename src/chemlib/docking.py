"""chemlib.docking - Receptor and ligand preparation, AutoDock Vina runner, pose extraction, and RMSD evaluation."""

from __future__ import annotations
import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


def prepare_ligand_3d(smiles: str, seed: int = 42) -> Optional[Chem.Mol]:
    """Generate an energy-minimized 3D conformer with hydrogens from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        res = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if res != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    return mol


def prepare_ligand_pdbqt_meeko(mol: Chem.Mol, out_path: Optional[str] = None) -> str:
    """Prepare a PDBQT string or file from a 3D RDKit Mol using Meeko."""
    from meeko import MoleculePreparation
    preparator = MoleculePreparation()
    mol_setup = preparator.prepare(mol)
    pdbqt_str = preparator.write_pdbqt_string(mol_setup)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pdbqt_str)
    return pdbqt_str


def get_box_from_ligand_mol(mol: Chem.Mol, padding: float = 8.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate center (x, y, z) and box sizes (size_x, size_y, size_z) from a reference ligand conformer."""
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)

    center = tuple((min_coords + max_coords) / 2.0)
    sizes = tuple((max_coords - min_coords) + 2 * padding)
    return center, sizes


def run_vina_docking(
    receptor_pdbqt_path: str,
    ligand_pdbqt_str_or_path: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    n_poses: int = 9,
    energy_range: float = 3.0,
    seed: int = 42,
) -> Tuple[List[float], str]:
    """Run AutoDock Vina docking via Python API.
    Returns:
    - affinities: list of binding energies (kcal/mol) for each pose
    - poses_pdbqt_str: PDBQT string containing all output poses
    """
    from vina import Vina

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt_path)

    # If ligand is a string buffer, write to a temp file
    is_path = os.path.exists(ligand_pdbqt_str_or_path)
    if not is_path:
        with tempfile.NamedTemporaryFile("w", suffix=".pdbqt", delete=False) as tmp:
            tmp.write(ligand_pdbqt_str_or_path)
            lig_path = tmp.name
    else:
        lig_path = ligand_pdbqt_str_or_path

    try:
        v.set_ligand_from_file(lig_path)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        affinities = [float(e) for e in v.energies(n_poses=n_poses)[:, 0]]
        poses_pdbqt_str = v.poses(n_poses=n_poses)
        return affinities, poses_pdbqt_str
    finally:
        if not is_path and os.path.exists(lig_path):
            os.remove(lig_path)


def compute_heavy_atom_rmsd(ref_mol: Chem.Mol, query_mol: Chem.Mol) -> float:
    """Compute symmetry-aware heavy atom RMSD between reference and query conformers."""
    ref_no_h = Chem.RemoveHs(ref_mol)
    query_no_h = Chem.RemoveHs(query_mol)
    return float(rdMolAlign.CalcRMS(query_no_h, ref_no_h))
