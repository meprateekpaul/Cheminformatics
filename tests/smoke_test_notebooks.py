#!/usr/bin/env python3
"""
smoke_test_notebooks.py
Executes a representative cross-section of solution notebooks across all curriculum parts
using nbclient to verify full runtime execution and assertion passes.
"""

import os
import sys
import time

import nbformat
from nbclient import NotebookClient


def test_notebook(notebook_path):
    print(f"\n{'=' * 60}")
    print(f"Executing: {os.path.basename(notebook_path)}")
    print(f"{'=' * 60}")
    start = time.time()
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute(cwd=os.path.dirname(notebook_path))
    duration = time.time() - start
    print(f"PASSED in {duration:.2f}s: {os.path.basename(notebook_path)}")
    return True


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sol_dir = os.path.join(repo_root, "solutions")

    # Test one key notebook from each pedagogical part (A through G)
    test_suite = [
        "nb00_setup_and_refresher.ipynb",  # Part A: Foundations
        "nb03_smiles_and_linear_notations.ipynb",  # Part A: SMILES & InChI
        "nb04_molecular_graphs_and_matrices.ipynb",  # Part A: Graph matrices
        "nb05_file_formats_and_conversion.ipynb",  # Part A: 3D Conversions
        "nb06_chemical_databases_chembl_pubchem.ipynb",  # Part B: Curation
        "nb09_fingerprints_keys_paths_circles.ipynb",  # Part C: Fingerprints
        "nb10_substructure_and_similarity_search.ipynb",  # Part D: LBVS & Searching
        "nb12_qsar_foundations_hansch.ipynb",  # Part E: QSAR
        "nb15_druglikeness_and_admet.ipynb",  # Part E: ADMET
        "nb16_target_discovery_and_prep.ipynb",  # Part F: Structure-Based
        "nb20_project_template_ligand_model.ipynb",  # Part G: Capstone
    ]

    passed = []
    failed = []

    for nb_name in test_suite:
        nb_path = os.path.join(sol_dir, nb_name)
        if not os.path.exists(nb_path):
            print(f"SKIPPED (not found): {nb_name}")
            continue
        try:
            test_notebook(nb_path)
            passed.append(nb_name)
        except Exception as e:
            print(f"FAILED: {nb_name}\nError: {e}")
            failed.append((nb_name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST SUMMARY: {len(passed)}/{len(test_suite)} PASSED")
    if failed:
        for name, err in failed:
            print(f"  ❌ {name}: {err}")
        sys.exit(1)
    else:
        print("All tested solution notebooks executed successfully to completion!")


if __name__ == "__main__":
    main()
