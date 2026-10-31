"""chemlib.io - Reading, writing, and format conversions for molecular files."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def read_smiles_file(
    filepath: str, delimiter: str = "\t", smiles_col: int = 0, name_col: int = 1
) -> list[tuple[Chem.Mol | None, str]]:
    """Read a .smi or .txt file containing SMILES and optional molecule names."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            smi = parts[smiles_col].strip()
            name = parts[name_col].strip() if len(parts) > name_col else ""
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and name:
                mol.SetProp("_Name", name)
            results.append((mol, name or smi))
    return results


def write_smiles_file(
    mols: list[Chem.Mol], filepath: str, delimiter: str = "\t", include_name: bool = True
) -> None:
    """Write a list of molecules to a .smi file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for mol in mols:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") and include_name else ""
            if name:
                f.write(f"{smi}{delimiter}{name}\n")
            else:
                f.write(f"{smi}\n")


def read_sdf(filepath: str, sanitize: bool = True, remove_hs: bool = True) -> list[Chem.Mol]:
    """Read an SD file into a list of RDKit molecules."""
    suppl = Chem.SDMolSupplier(filepath, sanitize=sanitize, removeHs=remove_hs)
    return [mol for mol in suppl if mol is not None]


def write_sdf(mols: list[Chem.Mol], filepath: str) -> None:
    """Write a list of molecules to an SD file."""
    writer = Chem.SDWriter(filepath)
    for mol in mols:
        if mol is not None:
            writer.write(mol)
    writer.close()


def sdf_to_dataframe(
    filepath: str, smiles_col: str = "SMILES", mol_col: str = "ROMol"
) -> pd.DataFrame:
    """Load an SDF directly into a pandas DataFrame with RDKit molecule representations."""
    df = PandasTools.LoadSDF(filepath, smilesName=smiles_col, molColName=mol_col)
    return df


def dataframe_to_sdf(
    df: pd.DataFrame, filepath: str, mol_col: str = "ROMol", properties: list[str] | None = None
) -> None:
    """Save a DataFrame containing RDKit molecules to an SDF."""
    PandasTools.WriteSDF(df, filepath, molColName=mol_col, properties=properties)


def smiles_to_mol2_obabel(smiles: str, name: str = "MOL") -> str:
    """Convert a SMILES string to a 3D Tripos MOL2 block using Open Babel."""
    try:
        from openbabel import openbabel as ob

        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("smi", "mol2")
        mol = ob.OBMol()
        if obConversion.ReadString(mol, smiles):
            mol.SetTitle(name)
            builder = ob.OBBuilder()
            builder.Build(mol)
            mol.AddHydrogens()
            ff = ob.OBForceField.FindForceField("mmff94")
            if ff and ff.Setup(mol):
                ff.ConjugateGradients(250)
                ff.GetCoordinates(mol)
            return obConversion.WriteString(mol)
    except Exception:
        pass

    cmd = ["obabel", f"-:{smiles}", "-omol2", "-h", "--gen3d", "--title", name]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def mol_to_mol2_file_obabel(mol: Chem.Mol, out_path: str) -> None:
    """Save an RDKit Mol to a MOL2 file using Open Babel."""
    try:
        from openbabel import openbabel as ob

        mol_block = Chem.MolToMolBlock(mol)
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("mol", "mol2")
        obmol = ob.OBMol()
        if obConversion.ReadString(obmol, mol_block):
            obConversion.WriteFile(obmol, out_path)
            return
    except Exception:
        pass

    with tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        write_sdf([mol], tmp_name)
        cmd = ["obabel", tmp_name, "-O", out_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def mol_to_pdbqt(mol: Chem.Mol, out_path: str | None = None) -> str:
    """Convert a 3D RDKit Mol to PDBQT format using Meeko."""
    try:
        from meeko import MoleculePreparation

        preparator = MoleculePreparation()
        mol_setup = preparator.prepare(mol)
        pdbqt_str = preparator.write_pdbqt_string(mol_setup)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pdbqt_str)
        return pdbqt_str
    except Exception:
        # Fallback to obabel if meeko fails
        with tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False) as tmp:
            tmp_name = tmp.name
        try:
            Chem.MolToMolFile(mol, tmp_name)
            cmd = ["obabel", tmp_name, "-opdbqt"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if out_path:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(res.stdout)
            return res.stdout
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
