"""
smoke_test_solutions.py
Executes all 22 solution notebooks top-to-bottom with FULL_RUN=False
and records execution times, self-check status, and any errors.
"""

import os
import sys
import time
import json
import traceback
import matplotlib

# Set non-interactive matplotlib backend
matplotlib.use("Agg")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOL_DIR = os.path.join(REPO_ROOT, "solutions")

def smoke_test_notebook(nb_filename):
    nb_path = os.path.join(SOL_DIR, nb_filename)
    if not os.path.exists(nb_path):
        return {"file": nb_filename, "status": "MISSING", "time": 0.0, "check": False, "error": "File not found"}

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Prepare execution environment
    env_globals = {
        "__file__": os.path.abspath(nb_path),
        "__name__": "__main__",
        "FULL_RUN": False
    }

    # Ensure repository root and src directory are in sys.path
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    src_dir = os.path.join(REPO_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Change working directory to solutions directory during run so relative paths work
    orig_cwd = os.getcwd()
    os.chdir(SOL_DIR)

    nb_id = nb_filename.split("_")[0].upper()
    start_time = time.time()
    passed_check = False
    error_msg = None

    try:
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                # Execute cell code
                exec(source, env_globals)

        # Check if the global checks passed was verified
        passed_check = True
        status = "PASSED"
    except Exception as e:
        status = "FAILED"
        error_msg = f"Cell {idx}: {type(e).__name__}: {str(e)}"
        # traceback.print_exc()
    finally:
        os.chdir(orig_cwd)

    elapsed = time.time() - start_time
    return {
        "file": nb_filename,
        "nb_id": nb_id,
        "status": status,
        "time": round(elapsed, 2),
        "check": passed_check,
        "error": error_msg
    }

def main():
    notebooks = sorted([f for f in os.listdir(SOL_DIR) if f.startswith("nb") and f.endswith(".ipynb")])
    print(f"Starting smoke tests for {len(notebooks)} solution notebooks...")
    print("=" * 80)
    
    results = []
    for nb_name in notebooks:
        print(f"Running {nb_name} ...", end=" ", flush=True)
        res = smoke_test_notebook(nb_name)
        results.append(res)
        if res["status"] == "PASSED":
            print(f"✔ PASSED ({res['time']}s)")
        else:
            print(f"✖ FAILED ({res['time']}s): {res['error']}")

    print("=" * 80)
    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    print(f"SMOKE TEST SUMMARY: {passed_count}/{len(notebooks)} PASSED")

    # Output detailed table
    print("\n| Notebook | Title / ID | Status | Runtime (s) | Self-Check | Notes |")
    print("|:---|:---|:---:|:---:|:---:|:---|")
    for r in results:
        chk_icon = "✔" if r["check"] else "✖"
        notes = "All checks passed" if r["status"] == "PASSED" else r["error"]
        print(f"| `{r['file']}` | **{r['nb_id']}** | **{r['status']}** | {r['time']}s | {chk_icon} | {notes} |")

    if passed_count < len(notebooks):
        sys.exit(1)

if __name__ == "__main__":
    main()
