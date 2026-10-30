import json
import os
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_notebook():
    return {
        "cells": [],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def add_md(nb, source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(nb, source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

def save_notebooks(nb_template, student_relpath, solution_relpath, alias_relpaths=None):
    student_nb = {
        "cells": [],
        "metadata": nb_template["metadata"],
        "nbformat": nb_template["nbformat"],
        "nbformat_minor": nb_template["nbformat_minor"]
    }
    solution_nb = {
        "cells": [],
        "metadata": nb_template["metadata"],
        "nbformat": nb_template["nbformat"],
        "nbformat_minor": nb_template["nbformat_minor"]
    }

    for cell in nb_template["cells"]:
        if cell["cell_type"] == "markdown":
            student_nb["cells"].append(cell)
            solution_nb["cells"].append(cell)
        else:
            raw_text = "".join(cell["source"])
            
            # Check for solution marker
            if "# --- SOLUTION BEGIN ---" in raw_text:
                # Student version: replace between markers with stub
                before, rest = raw_text.split("# --- SOLUTION BEGIN ---", 1)
                inside, after = rest.split("# --- SOLUTION END ---", 1)
                student_text = before + "# YOUR CODE HERE\nraise NotImplementedError(\"Student implementation here\")\n" + after
                
                # Solution version: keep clean solution without markers
                sol_text = before + inside.strip("\n") + "\n" + after
                
                st_lines = [l + "\n" for l in student_text.strip().split("\n")]
                if st_lines: st_lines[-1] = st_lines[-1].rstrip("\n")
                student_nb["cells"].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": cell.get("metadata", {}),
                    "outputs": [],
                    "source": st_lines
                })
                
                sol_lines = [l + "\n" for l in sol_text.strip().split("\n")]
                if sol_lines: sol_lines[-1] = sol_lines[-1].rstrip("\n")
                solution_nb["cells"].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": cell.get("metadata", {}),
                    "outputs": [],
                    "source": sol_lines
                })
            else:
                student_nb["cells"].append(cell)
                solution_nb["cells"].append(cell)

    student_path = os.path.join(REPO_ROOT, student_relpath)
    solution_path = os.path.join(REPO_ROOT, solution_relpath)
    os.makedirs(os.path.dirname(student_path), exist_ok=True)
    os.makedirs(os.path.dirname(solution_path), exist_ok=True)

    with open(student_path, "w", encoding="utf-8") as f:
        json.dump(student_nb, f, indent=1)
    with open(solution_path, "w", encoding="utf-8") as f:
        json.dump(solution_nb, f, indent=1)
    
    print(f"Saved:\n  Student:  {student_path}\n  Solution: {solution_path}")

    if alias_relpaths:
        for st_alias, sol_alias in alias_relpaths:
            st_a_path = os.path.join(REPO_ROOT, st_alias)
            sol_a_path = os.path.join(REPO_ROOT, sol_alias)
            shutil.copyfile(student_path, st_a_path)
            shutil.copyfile(solution_path, sol_a_path)
            print(f"  Alias student:  {st_a_path}\n  Alias solution: {sol_a_path}")
