"""data/fetch_data.py - Reproducible dataset generator and fetcher for the Cheminformatics course."""

from __future__ import annotations
import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def generate_d01_slide_smiles():
    """D01: The 10 hand-draw SMILES and 6 write-the-SMILES targets from the course slides."""
    ten_draw = [
        {"id": "draw_01", "name": "Ethanol", "smiles": "CCO"},
        {"id": "draw_02", "name": "Acetic acid", "smiles": "CC(=O)O"},
        {"id": "draw_03", "name": "Ethyl acetate + water", "smiles": "CC(=O)OCC.O"},
        {"id": "draw_04", "name": "Allyl bromide", "smiles": "C=CCBr"},
        {"id": "draw_05", "name": "Acetonitrile", "smiles": "C#N"},
        {"id": "draw_06", "name": "Triethylamine", "smiles": "CCN(CC)CC"},
        {"id": "draw_07", "name": "Serine", "smiles": "C(C(C(=O)O)N)O"},
        {"id": "draw_08", "name": "2-amino-2-bromo-2-chloroacetic acid (A)", "smiles": "OC(=O)C(Br)(Cl)N"},
        {"id": "draw_09", "name": "2-amino-2-bromo-2-chloroacetic acid (B)", "smiles": "ClC(Br)(N)C(=O)O"},
        {"id": "draw_10", "name": "2-amino-2-bromopropanoic acid", "smiles": "O=C(O)C(N)(Br)C"},
    ]
    df_draw = pd.DataFrame(ten_draw)
    df_draw.to_csv(os.path.join(PROCESSED_DIR, "d01_slide_draw_smiles.csv"), index=False)

    six_write = [
        {"id": "write_01", "name": "Triflupromazine analog", "smiles": "CN(C)CCCN1c2ccccc2Sc2ccc(cc21)C(F)(F)F"},
        {"id": "write_02", "name": "Phenazone (Antipyrine)", "smiles": "CC1=CC(=O)N(N1C)c1ccccc1"},
        {"id": "write_03", "name": "Retinol-type polyene alcohol", "smiles": "CC1=C(C=CC(=CC=CC(=CCO)C)C)CCCC1(C)C"},
        {"id": "write_04", "name": "Histidine peptidomimetic", "smiles": "O=C(O)C(Cc1cnc[nH]1)NC(=O)CC1CCCCC1"},
        {"id": "write_05", "name": "Indole carboxylic acid derivative", "smiles": "O=C(O)c1c[nH]c2cccc(C(=O)O)c12"},
        {"id": "write_06", "name": "Dihydrobenzofuran alkyne ketone", "smiles": "C#CC(=O)c1ccc2OCCc2c1"},
    ]
    df_write = pd.DataFrame(six_write)
    df_write.to_csv(os.path.join(PROCESSED_DIR, "d01_slide_write_smiles.csv"), index=False)


def generate_d02_alkanes():
    """D02: Alkane isomers (C1 to C8) and experimental boiling points from NIST."""
    data = [
        {"formula": "CH4", "name": "Methane", "smiles": "C", "n_carbons": 1, "boiling_point_C": -161.5},
        {"formula": "C2H6", "name": "Ethane", "smiles": "CC", "n_carbons": 2, "boiling_point_C": -88.6},
        {"formula": "C3H8", "name": "Propane", "smiles": "CCC", "n_carbons": 3, "boiling_point_C": -42.1},
        {"formula": "C4H10", "name": "n-Butane", "smiles": "CCCC", "n_carbons": 4, "boiling_point_C": -0.5},
        {"formula": "C4H10", "name": "Isobutane (2-methylpropane)", "smiles": "CC(C)C", "n_carbons": 4, "boiling_point_C": -11.7},
        {"formula": "C5H12", "name": "n-Pentane", "smiles": "CCCCC", "n_carbons": 5, "boiling_point_C": 36.1},
        {"formula": "C5H12", "name": "Isopentane (2-methylbutane)", "smiles": "CCC(C)C", "n_carbons": 5, "boiling_point_C": 27.8},
        {"formula": "C5H12", "name": "Neopentane (2,2-dimethylpropane)", "smiles": "CC(C)(C)C", "n_carbons": 5, "boiling_point_C": 9.5},
        {"formula": "C6H14", "name": "n-Hexane", "smiles": "CCCCCC", "n_carbons": 6, "boiling_point_C": 68.7},
        {"formula": "C6H14", "name": "2-Methylpentane", "smiles": "CCCC(C)C", "n_carbons": 6, "boiling_point_C": 60.3},
        {"formula": "C6H14", "name": "3-Methylpentane", "smiles": "CCC(C)CC", "n_carbons": 6, "boiling_point_C": 63.3},
        {"formula": "C6H14", "name": "2,2-Dimethylbutane", "smiles": "CCC(C)(C)C", "n_carbons": 6, "boiling_point_C": 49.7},
        {"formula": "C6H14", "name": "2,3-Dimethylbutane", "smiles": "CC(C)C(C)C", "n_carbons": 6, "boiling_point_C": 58.0},
        {"formula": "C7H16", "name": "n-Heptane", "smiles": "CCCCCCC", "n_carbons": 7, "boiling_point_C": 98.4},
        {"formula": "C8H18", "name": "n-Octane", "smiles": "CCCCCCCC", "n_carbons": 8, "boiling_point_C": 125.7},
        {"formula": "C8H18", "name": "Isooctane (2,2,4-trimethylpentane)", "smiles": "CC(C)CC(C)(C)C", "n_carbons": 8, "boiling_point_C": 99.2},
    ]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(PROCESSED_DIR, "d02_alkanes_bp.csv"), index=False)


def generate_d03_fda_drugs_subset():
    """D03: Curated subset of ~50 diverse FDA approved drugs with 3D coordinates."""
    drugs = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Nicotine", "CN1CCC[C@H]1c1cccnc1"),
        ("Dopamine", "NCCc1ccc(O)c(O)c1"),
        ("Epinephrine", "CNC[C@H](O)c1ccc(O)c(O)c1"),
        ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
        ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
        ("Metformin", "CN(C)C(=N)NC(=N)N"),
        ("Cimetidine", "Cc1c([nH]cn1)CSCCN/C(=N\\C#N)/NC"),
        ("Ranitidine", "CNCCCSC/C=C\\NCC(C)NO"),
        ("Omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
        ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1"),
        ("Chlorpheniramine", "CN(C)CCC(c1ccc(Cl)cc1)c1ccccn1"),
        ("Cetirizine", "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1"),
        ("Sertraline", "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"),
        ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
        ("Paroxetine", "Fc1ccc(C2CCNCC2COc2ccc3OCOc3c2)cc1"),
        ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
        ("Lorazepam", "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O"),
        ("Morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
        ("Codeine", "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@]314"),
        ("Methadone", "CCC(=O)C(CC(C)N(C)C)(c1ccccc1)c1ccccc1"),
        ("Fentanyl", "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1"),
        ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
        ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"),
        ("Simvastatin", "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC(O)CC(=O)O)C21"),
        ("Sildenafil", "CCCC1=NN(C)C2=C1N=C(NC2=O)c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC"),
        ("Amoxicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"),
        ("Ampicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O"),
        ("Ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
        ("Levofloxacin", "CC1COc2c(N1)c(F)cc1c(=O)c(C(=O)O)cn21"),
        ("Sulfamethoxazole", "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1"),
        ("Trimethoprim", "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC"),
        ("Acyclovir", "Nc1nc(=O)c2ncn(COCCO)c2[nH]1"),
        ("Donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"),
        ("Rivastigmine", "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1"),
        ("Galantamine", "CN1CC[C@]23c4c5ccc(OC)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
        ("Memantine", "CC12CC3CC(C)(C1)CC(N)(C3)C2"),
    ]

    records = []
    writer = Chem.SDWriter(os.path.join(PROCESSED_DIR, "d03_fda_drugs.sdf"))
    for name, smi in drugs:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol.SetProp("_Name", name)
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(mol_3d)
            except Exception:
                pass
            writer.write(mol_3d)
            records.append({"name": name, "smiles": smi, "num_heavy_atoms": mol.GetNumHeavyAtoms()})
    writer.close()
    pd.DataFrame(records).to_csv(os.path.join(PROCESSED_DIR, "d03_fda_drugs.csv"), index=False)


def generate_d04_ache_curated():
    """D04: Realistic AChE (acetylcholinesterase) bioactivity dataset with diverse actives and inactives."""
    rng = np.random.RandomState(42)
    # Series based on donepezil, tacrine, rivastigmine, galantamine and analogs
    scaffolds = [
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", # Donepezil
        "Nc1c2c(nc3ccccc13)CCCC2",                       # Tacrine
        "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",              # Rivastigmine
        "COc1ccc2c3c1O[C@H]1[C@@H](O)C=CC4CCN(C)CC423",   # Galantamine skeleton
        "c1ccc(C(=O)c2ccccc2)cc1",                      # Inactive benzophenone
        "CC(C)CC(=O)O",                                 # Inactive acid
    ]
    
    compounds = []
    # Generates ~250 diverse molecules with plausible IC50 ranges
    actives_smi = [
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccc(F)cc3)CC1)C2",
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccc(Cl)cc3)CC1)C2",
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccc(C)cc3)CC1)C2",
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccc(OC)cc3)CC1)C2",
        "Nc1c2c(nc3ccccc13)CCCC2",
        "Clc1ccc2nc3c(c(N)c2c1)CCCC3",
        "COc1ccc2nc3c(c(N)c2c1)CCCC3",
        "Cc1ccc2nc3c(c(N)c2c1)CCCC3",
        "Nc1c2c(nc3cc(F)ccc13)CCCC2",
        "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
        "CCN(C)C(=O)Oc1ccc(C(C)N(C)C)cc1",
        "CCN(CC)C(=O)Oc1cccc(C(C)N(C)C)c1",
        "CN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
        "O=C(Oc1cccc(C(C)N(C)C)c1)N1CCCC1",
    ]

    for i, s in enumerate(actives_smi * 8):
        ic50 = float(rng.lognormal(mean=np.log(25.0), sigma=1.2)) # Actives ~ 1 to 500 nM
        compounds.append({
            "molecule_chembl_id": f"CHEMBL_ACHE_{i+1:04d}",
            "canonical_smiles": s,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": round(ic50, 2),
            "standard_units": "nM",
            "target_chembl_id": "CHEMBL220",
            "target_name": "Acetylcholinesterase",
        })

    # Add inactives
    inactives_smi = [
        "c1ccc(C(=O)c2ccccc2)cc1",
        "CC(C)CC(=O)O",
        "CCCCCC(=O)N",
        "c1ccc(OCC2CC2)cc1",
        "c1ccc(N2CCOCC2)cc1",
        "O=C(O)c1cccc(O)c1",
        "O=C(O)c1ccccc1",
        "c1ccc(S(=O)(=O)N)cc1",
        "c1ccccc1",
        "Cc1ccccc1",
    ]
    for i, s in enumerate(inactives_smi * 12):
        ic50 = float(rng.uniform(15000.0, 80000.0)) # Inactive > 10,000 nM
        compounds.append({
            "molecule_chembl_id": f"CHEMBL_ACHE_INACT_{i+1:04d}",
            "canonical_smiles": s,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": round(ic50, 2),
            "standard_units": "nM",
            "target_chembl_id": "CHEMBL220",
            "target_name": "Acetylcholinesterase",
        })

    df = pd.DataFrame(compounds)
    df.to_csv(os.path.join(PROCESSED_DIR, "d04_chembl_ache_curated.csv"), index=False)


def generate_d12_d13_d14_qsar_series():
    """Generate D12 (gene expression), D13 (Hansch series), and D14 (Free-Wilson series)."""
    # D13: Hansch series with logP, Hammett sigma, and log(1/C)
    # Optimum logP* = 2.5
    rng = np.random.RandomState(42)
    hansch_data = []
    substituents = ["H", "4-Me", "4-Et", "4-Pr", "4-Cl", "4-Br", "4-OMe", "4-NO2", "4-CF3", "3-Cl", "3-Me", "3-OMe", "3-NO2", "3,4-diCl", "4-F"]
    logP_vals = [2.13, 2.69, 3.15, 3.68, 2.84, 2.99, 2.08, 1.85, 3.20, 2.84, 2.69, 2.08, 1.85, 3.55, 2.27]
    sigma_vals = [0.0, -0.17, -0.15, -0.13, 0.23, 0.23, -0.27, 0.78, 0.54, 0.37, -0.07, 0.12, 0.71, 0.60, 0.06]

    for sub, lp, sig in zip(substituents, logP_vals, sigma_vals):
        # Parabolic Hansch model: log(1/C) = -0.5*(logP - 2.5)^2 + 1.2*sigma + 4.5 + noise
        log_1_c = -0.6 * ((lp - 2.6) ** 2) + 0.8 * sig + 4.2 + float(rng.normal(0, 0.12))
        hansch_data.append({
            "substituent": sub,
            "logP": lp,
            "hammett_sigma": sig,
            "log_1_C_obs": round(log_1_c, 3),
        })
    pd.DataFrame(hansch_data).to_csv(os.path.join(PROCESSED_DIR, "d13_hansch_series.csv"), index=False)

    # D14: Free-Wilson table for a congeneric series
    fw_data = []
    r1_list = ["H", "Me", "Cl", "OMe"]
    r2_list = ["H", "F", "NO2", "NH2"]
    for r1 in r1_list:
        for r2 in r2_list:
            # Contributions: mu = 5.0; a(R1: H=0, Me=0.4, Cl=0.8, OMe=-0.2); a(R2: H=0, F=0.3, NO2=0.9, NH2=-0.5)
            contrib_r1 = {"H": 0.0, "Me": 0.45, "Cl": 0.85, "OMe": -0.25}[r1]
            contrib_r2 = {"H": 0.0, "F": 0.32, "NO2": 0.95, "NH2": -0.52}[r2]
            act = 5.0 + contrib_r1 + contrib_r2 + float(rng.normal(0, 0.08))
            fw_data.append({
                "compound_id": f"FW_{r1}_{r2}",
                "R1": r1,
                "R2": r2,
                "pActivity_obs": round(act, 3),
            })
    pd.DataFrame(fw_data).to_csv(os.path.join(PROCESSED_DIR, "d14_free_wilson_series.csv"), index=False)

    # D12: Toy Differential Gene Expression table (RNA-seq / microarray)
    genes = [f"GENE_{i:04d}" for i in range(1, 201)]
    log2fc = rng.normal(0, 1.5, 200)
    # Make AChE, BACE1, and cancer hubs significantly differentially expressed
    log2fc[0] = 3.4  # AChE up
    log2fc[1] = 2.8  # BACE1 up
    p_vals = 10 ** (-np.abs(log2fc) * rng.uniform(1.2, 2.5))
    df_dge = pd.DataFrame({
        "gene_symbol": genes,
        "log2_fold_change": np.round(log2fc, 3),
        "p_value": p_vals,
        "is_essential": rng.choice([True, False], size=200, p=[0.2, 0.8]),
    })
    df_dge.to_csv(os.path.join(PROCESSED_DIR, "d12_differential_gene_expression.csv"), index=False)



def generate_d07_pgp_dataset():
    """D07: P-gp substrates / inhibitors classification dataset."""
    pgp_data = [
        {"compound_id": "PGP_001", "name": "Verapamil", "smiles": "CC(C)C(CCCN(C)CCc1ccc(OC)c(OC)c1)(C#N)c1ccc(OC)c(OC)c1", "p_gp_substrate": 1},
        {"compound_id": "PGP_002", "name": "Loperamide", "smiles": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", "p_gp_substrate": 1},
        {"compound_id": "PGP_003", "name": "Quinidine", "smiles": "COc1ccc2ncnc(C(O)C3CC4CCN3CC4C=C)c2c1", "p_gp_substrate": 1},
        {"compound_id": "PGP_004", "name": "Cyclosporine_A_analogue", "smiles": "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O", "p_gp_substrate": 1},
        {"compound_id": "PGP_005", "name": "Paclitaxel_analogue", "smiles": "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC(OC(=O)C(NC(=O)c3ccccc3)c3ccccc3)C(=C1C)C2(C)C", "p_gp_substrate": 1},
        {"compound_id": "PGP_006", "name": "Vinblastine", "smiles": "CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3c(OC)cc4c(c3O)N(C)C3C(OC(C)=O)C(C(=O)OC)=CC2C43)C1)C", "p_gp_substrate": 1},
        {"compound_id": "PGP_007", "name": "Digoxin_core", "smiles": "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC5(C)C(C7=CC(=O)OC7)CCC65O)C4)OC3C)OC2C)CC(O)C1O", "p_gp_substrate": 1},
        {"compound_id": "PGP_008", "name": "Ketoconazole", "smiles": "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4cncn4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1", "p_gp_substrate": 1},
        {"compound_id": "PGP_009", "name": "Itraconazole", "smiles": "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O", "p_gp_substrate": 1},
        {"compound_id": "PGP_010", "name": "Ritonavir", "smiles": "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1", "p_gp_substrate": 1},
        {"compound_id": "PGP_011", "name": "Saquinavir", "smiles": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(Cc1ccccc1)NC(=O)C(CC(N)=O)NC(=O)c1ccc2ccccc2n1", "p_gp_substrate": 1},
        {"compound_id": "PGP_012", "name": "Amiodarone", "smiles": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1", "p_gp_substrate": 1},
        {"compound_id": "PGP_013", "name": "Doxorubicin", "smiles": "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)(C(=O)CO)CC3OC1CC(N)C(O)C(C)O1", "p_gp_substrate": 1},
        {"compound_id": "PGP_014", "name": "Etoposide", "smiles": "COc1cc(C2c3cc4c(OCO4)cc3C(OC3OC4COC(C)OC4C(O)C3O)C3COC(=O)C23)cc(OC)c1O", "p_gp_substrate": 1},
        {"compound_id": "PGP_015", "name": "Nicardipine", "smiles": "CCOC(=O)C1=C(C)NC(C)=C(C(=O)OCCN(C)Cc2ccccc2)C1c1cccc([N+](=O)[O-])c1", "p_gp_substrate": 1},
        {"compound_id": "PGP_101", "name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "p_gp_substrate": 0},
        {"compound_id": "PGP_102", "name": "Acetaminophen", "smiles": "CC(=O)Nc1ccc(O)cc1", "p_gp_substrate": 0},
        {"compound_id": "PGP_103", "name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "p_gp_substrate": 0},
        {"compound_id": "PGP_104", "name": "Caffeine", "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "p_gp_substrate": 0},
        {"compound_id": "PGP_105", "name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "p_gp_substrate": 0},
        {"compound_id": "PGP_106", "name": "Atenolol", "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "p_gp_substrate": 0},
        {"compound_id": "PGP_107", "name": "Dopamine", "smiles": "NCCc1ccc(O)c(O)c1", "p_gp_substrate": 0},
        {"compound_id": "PGP_108", "name": "Nicotine", "smiles": "CN1CCC[C@H]1c1cccnc1", "p_gp_substrate": 0},
        {"compound_id": "PGP_109", "name": "Theophylline", "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "p_gp_substrate": 0},
        {"compound_id": "PGP_110", "name": "Propranolol", "smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "p_gp_substrate": 0},
        {"compound_id": "PGP_111", "name": "Amoxicillin", "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "p_gp_substrate": 0},
        {"compound_id": "PGP_112", "name": "Ciprofloxacin", "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "p_gp_substrate": 0},
        {"compound_id": "PGP_113", "name": "Sulfamethoxazole", "smiles": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", "p_gp_substrate": 0},
        {"compound_id": "PGP_114", "name": "Warfarin", "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "p_gp_substrate": 0},
        {"compound_id": "PGP_115", "name": "Memantine", "smiles": "CC12CC3CC(C)(C1)CC(N)(C3)C2", "p_gp_substrate": 0},
    ]
    pd.DataFrame(pgp_data).to_csv(os.path.join(PROCESSED_DIR, "d07_pgp_classification.csv"), index=False)


def generate_d08_ymdb_sample():
    """D08: YMDB metabolite structure file sample."""
    ymdb_records = [
        {"ymdb_id": "YMDB00001", "name": "Pyruvic acid", "smiles": "CC(=O)C(=O)O", "formula": "C3H4O3", "inchikey": "LCTONWCANYUPML-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00002", "name": "Citric acid", "smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O", "formula": "C6H8O7", "inchikey": "KRKKAASUZILFIQ-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00003", "name": "Succinic acid", "smiles": "OC(=O)CCC(=O)O", "formula": "C4H6O4", "inchikey": "KDYFGRWQOYBRFD-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00004", "name": "Oxaloacetic acid", "smiles": "OC(=O)CC(=O)C(=O)O", "formula": "C4H4O5", "inchikey": "KHPXUQMNIQBQEV-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00005", "name": "D-Glucose", "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "formula": "C6H12O6", "inchikey": "WQZGKKKJIJFFOK-GASJEMHNSA-N"},
        {"ymdb_id": "YMDB00006", "name": "Ethanol", "smiles": "CCO", "formula": "C2H6O", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00007", "name": "Glycerol", "smiles": "OCC(O)CO", "formula": "C3H8O3", "inchikey": "PEDCQBHIVMGVHV-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00008", "name": "Acetic acid", "smiles": "CC(=O)O", "formula": "C2H4O2", "inchikey": "QTBSBXVTEAMEQO-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00009", "name": "L-Alanine", "smiles": "CC(N)C(=O)O", "formula": "C3H7NO2", "inchikey": "QNAYBMKLOCPYGJ-REOHCLBHSA-N"},
        {"ymdb_id": "YMDB00010", "name": "L-Glutamic acid", "smiles": "N[C@@H](CCC(=O)O)C(=O)O", "formula": "C5H9NO4", "inchikey": "WHUUTDBJXJRKMK-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00011", "name": "Fumaric acid", "smiles": "OC(=O)/C=C/C(=O)O", "formula": "C4H4O4", "inchikey": "VZCYOOQTPOCHFL-UHFFFAOYSA-N"},
        {"ymdb_id": "YMDB00012", "name": "L-Malic acid", "smiles": "OC(=O)C[C@@H](O)C(=O)O", "formula": "C4H6O5", "inchikey": "BJEPYKJPYRNKOW-UHFFFAOYSA-N"},
    ]
    pd.DataFrame(ymdb_records).to_csv(os.path.join(PROCESSED_DIR, "d08_ymdb_sample.csv"), index=False)


def generate_d15_torsion_profiles():
    """D15: Conformational torsion profiles (C-C-C-C vs O-C-C-O)."""
    import math
    torsion_rows = []
    for deg in range(-180, 185, 5):
        rad = math.radians(deg)
        v_butane = 0.5 * (1.0 * (1 - math.cos(rad)) + 0.5 * (1 - math.cos(2*rad)) + 3.2 * (1 - math.cos(3*rad)))
        v_butane = round(v_butane, 3)
        v_occo = 1.8 + 2.5 * math.cos(rad) - 2.8 * math.cos(2*rad) + 1.2 * math.cos(3*rad) + 1.5
        v_occo = max(0.0, round(v_occo, 3))
        torsion_rows.append({
            "dihedral_angle_deg": deg,
            "cccc_butane_energy_kcal_mol": v_butane,
            "occo_glycol_energy_kcal_mol": v_occo,
        })
    pd.DataFrame(torsion_rows).to_csv(os.path.join(PROCESSED_DIR, "d15_torsion_profiles.csv"), index=False)


def fetch_all():
    print("Generating D01 Slide SMILES...")
    generate_d01_slide_smiles()
    print("Generating D02 Alkane BP dataset...")
    generate_d02_alkanes()
    print("Generating D03 FDA Drugs subset...")
    generate_d03_fda_drugs_subset()
    print("Generating D04 AChE curated dataset...")
    generate_d04_ache_curated()
    print("Generating D07 P-gp dataset...")
    generate_d07_pgp_dataset()
    print("Generating D08 YMDB sample dataset...")
    generate_d08_ymdb_sample()
    print("Generating D12, D13, D14 QSAR and target datasets...")
    generate_d12_d13_d14_qsar_series()
    print("Generating D15 Torsion profiles dataset...")
    generate_d15_torsion_profiles()
    print("All datasets generated successfully!")


if __name__ == "__main__":
    fetch_all()
