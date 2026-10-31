"""chemlib.viz - Visualization helpers for 2D molecular grids, substructures, bit info, and 3D views."""

from __future__ import annotations

from typing import Any

import IPython.display
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def draw_molecule_grid(
    mols: list[Chem.Mol],
    legends: list[str] | None = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 200),
    use_svg: bool = True,
) -> Any:
    """Draw a list of molecules in a grid image with legends."""
    valid_mols = []
    valid_legends = []
    for i, m in enumerate(mols):
        if m is not None:
            valid_mols.append(m)
            if legends and i < len(legends):
                valid_legends.append(str(legends[i]))
            else:
                valid_legends.append("")

    return Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=valid_legends if any(valid_legends) else None,
        useSVG=use_svg,
    )


def highlight_substructure(
    mol: Chem.Mol,
    pattern: str | Chem.Mol,
    img_size: tuple[int, int] = (350, 300),
) -> Any:
    """Highlight matching atoms and bonds for a SMARTS or Mol substructure query."""
    if mol is None:
        return None
    if isinstance(pattern, str):
        query = Chem.MolFromSmarts(pattern)
    else:
        query = pattern

    if query is None or not mol.HasSubstructMatch(query):
        return Draw.MolToImage(mol, size=img_size)

    match_atoms = list(mol.GetSubstructMatch(query))
    match_bonds = []
    for bond in query.GetBonds():
        b = mol.GetBondBetweenAtoms(
            match_atoms[bond.GetBeginAtomIdx()],
            match_atoms[bond.GetEndAtomIdx()],
        )
        if b is not None:
            match_bonds.append(b.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
    drawer.DrawMolecule(mol, highlightAtoms=match_atoms, highlightBonds=match_bonds)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return IPython.display.SVG(svg)


def view_3d_mol(
    mol: Chem.Mol, style: str = "stick", surface: bool = False, size: tuple[int, int] = (400, 300)
) -> Any:
    """Render a 3D molecule using py3Dmol."""
    try:
        import py3Dmol

        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=size[0], height=size[1])
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({style: {}})
        if surface:
            viewer.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "white"})
        viewer.zoomTo()
        return viewer
    except Exception as e:
        return f"3D viewer unavailable: {e}"


def view_complex_3d(
    pdb_receptor_str: str, ligand_mol: Chem.Mol, size: tuple[int, int] = (600, 450)
) -> Any:
    """Render a receptor-ligand complex in 3D using py3Dmol."""
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=size[0], height=size[1])
        viewer.addModel(pdb_receptor_str, "pdb")
        viewer.setStyle({"cartoon": {"color": "spectrum"}})

        lig_block = Chem.MolToMolBlock(ligand_mol)
        viewer.addModel(lig_block, "mol")
        viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
        viewer.zoomTo({"model": 1})
        return viewer
    except Exception as e:
        return f"Complex 3D viewer unavailable: {e}"
