import re
import os

nb_data = [
    {
        "id": "nb00",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Beginner",
        "diff_class": "bg-emerald-50 text-emerald-700 border-emerald-200",
        "dots_time": "●○○ · ~30 min",
        "title": "Setup, Python and RDKit Refresher",
        "desc": "Environment verification, reading/writing molecules, atom/bond iteration, implicit vs explicit H, PandasTools, and Open Babel CLI.",
        "topics": ["Environment", "RDKit Mol", "PandasTools", "Open Babel CLI"],
        "filename": "nb00_setup_and_refresher.ipynb"
    },
    {
        "id": "nb01",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Beginner",
        "diff_class": "bg-emerald-50 text-emerald-700 border-emerald-200",
        "dots_time": "●○○ · ~45 min",
        "title": "Informatics, Chemical Space and Drug Discovery",
        "desc": "Chemical space estimation (~10^60 compounds), alkane isomer enumeration (C1–C8), SBDD vs LBDD decision trees, and ΔG = RT ln Ki conversions.",
        "topics": ["Chemical Space", "Isomers", "SBDD vs LBDD", "Affinity Math"],
        "filename": "nb01_chemical_space_and_drug_discovery.ipynb"
    },
    {
        "id": "nb02",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~60 min",
        "title": "Building 3D Structures: Cartesian, Z-Matrix & Editors",
        "desc": "Constructing methane from geometry, orientation dependence of Cartesian coordinates, Z-matrix internal coordinate conversion, and py3Dmol conformer scans.",
        "topics": ["Cartesian (x,y,z)", "Z-Matrix", "py3Dmol", "Conformers"],
        "filename": "nb02_building_3d_structures.ipynb"
    },
    {
        "id": "nb03",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~75 min",
        "title": "SMILES and Linear Chemical Notations",
        "desc": "SMILES grammar, hand-drawing exercises (10 molecules), stereochemistry (@/@@, /,\\), InChI layers, and canonicalization differences across toolkits.",
        "topics": ["SMILES Grammar", "Stereo Flags", "InChI Layers", "Canonicalization"],
        "filename": "nb03_smiles_and_linear_notations.ipynb"
    },
    {
        "id": "nb04",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~75 min",
        "title": "Molecular Graphs and Matrices",
        "desc": "Graph representations, adjacency, bond-order, and topological distance matrices. Molecular walks (MWC) and self-returning walks (SRW). VF2 isomorphism.",
        "topics": ["Graph Theory", "Adjacency Matrix", "Distance Matrix", "VF2 Algorithm"],
        "filename": "nb04_molecular_graphs_and_matrices.ipynb"
    },
    {
        "id": "nb05",
        "part_label": "Part A · Foundations",
        "cat": "foundations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~75 min",
        "title": "File Formats and Conversion (2D ⇄ 3D)",
        "desc": "SDF, MOL, PDB, MOL2 formats. RDKit ETKDGv3 conformer generation, MMFF94 force field optimization, and Open Babel 3D coordinate conversion.",
        "topics": ["SDF / MOL / PDB", "ETKDGv3", "MMFF94", "Open Babel"],
        "filename": "nb05_file_formats_and_conversion.ipynb"
    },
    {
        "id": "nb06",
        "part_label": "Part B · Representations",
        "cat": "representations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~90 min",
        "title": "Chemical Databases: PubChem, ChEMBL, ZINC",
        "desc": "Programmatic REST queries to ChEMBL and PubChemPy, bioactivity schema, duplicate resolution, salt stripping, and curation pipeline.",
        "topics": ["ChEMBL REST", "PubChemPy", "Data Curation", "Salt Stripping"],
        "filename": "nb06_chemical_databases_chembl_pubchem.ipynb"
    },
    {
        "id": "nb07",
        "part_label": "Part B · Representations",
        "cat": "representations",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~75 min",
        "title": "Organic Structures, Isomers and Functional Groups",
        "desc": "Constitutional vs stereoisomers, Hückel 4n+2 aromaticity, SMARTS functional group detection, and Checkmol 204-key structural classifier.",
        "topics": ["Isomerism", "Hückel Rule", "SMARTS Patterns", "Checkmol Keys"],
        "filename": "nb07_organic_structures_and_functional_groups.ipynb"
    },
    {
        "id": "nb08",
        "part_label": "Part C · Descriptors",
        "cat": "descriptors",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~120 min",
        "title": "Molecular Descriptors (0D–4D)",
        "desc": "Topological indices (Wiener W, Zagreb M1/M2, Randić chi), correlation with alkane boiling points, 3D shape descriptors, and Mordred/PaDEL.",
        "topics": ["Wiener Index", "Zagreb Indices", "Randić Index", "Mordred 2D/3D"],
        "filename": "nb08_molecular_descriptors.ipynb"
    },
    {
        "id": "nb09",
        "part_label": "Part C · Descriptors",
        "cat": "descriptors",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~90 min",
        "title": "Fingerprints: Keys, Paths, Circles & Pharmacophores",
        "desc": "Structural keys (MACCS 166), topological paths (Daylight-like), circular Morgan/ECFP4/6 via modern rdFingerprintGenerator, and bit folding.",
        "topics": ["MACCS Keys", "ECFP4 / ECFP6", "Morgan Generator", "Bit Folding"],
        "filename": "nb09_fingerprints_keys_paths_circles.ipynb"
    },
    {
        "id": "nb10",
        "part_label": "Part D · Searching",
        "cat": "searching",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~120 min",
        "title": "Substructure and Similarity Search",
        "desc": "Hybrid screening (bit pre-filter + VF2 verification), 51 similarity coefficients implementation (Tanimoto, Dice, Cosine, Russell-Rao, Forbes), and LBVS benchmarks.",
        "topics": ["Substructure Search", "51 Coefficients", "Tanimoto", "Tversky & Dice"],
        "filename": "nb10_substructure_and_similarity_search.ipynb"
    },
    {
        "id": "nb11",
        "part_label": "Part D · Searching",
        "cat": "searching",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~120 min",
        "title": "Pharmacophore Modeling and Search",
        "desc": "Feature extraction (HBD, HBA, aromatic, positive/negative ionizable), 3-point pharmacophore distance triangles, and virtual screening enrichment.",
        "topics": ["ChemicalFeatures", "3-Point Triangles", "Pharmacophore Align", "Enrichment"],
        "filename": "nb11_pharmacophore_modeling.ipynb"
    },
    {
        "id": "nb12",
        "part_label": "Part E · QSAR & ML",
        "cat": "qsar_ml",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~90 min",
        "title": "QSAR Foundations: From Crum-Brown to Hansch",
        "desc": "Hammett sigma electronic constants, Taft steric Es, Hansch parabolic logP equations, Free-Wilson additivity matrix, and OLS regression.",
        "topics": ["Hansch Analysis", "Hammett Sigma", "Free-Wilson", "OLS Regression"],
        "filename": "nb12_qsar_foundations_hansch.ipynb"
    },
    {
        "id": "nb13",
        "part_label": "Part E · QSAR & ML",
        "cat": "qsar_ml",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~150 min",
        "title": "Machine Learning for QSAR (Classification & Regression)",
        "desc": "Random Forest, SVM, kNN, Bemis-Murcko scaffold split vs random split, LOO cross-validation, Matthews Correlation Coefficient (MCC), and ROC-AUC.",
        "topics": ["Random Forest", "Scaffold Split", "MCC / ROC-AUC", "Applicability Domain"],
        "filename": "nb13_ml_qsar_classification_regression.ipynb"
    },
    {
        "id": "nb14",
        "part_label": "Part E · QSAR & ML",
        "cat": "qsar_ml",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~120 min",
        "title": "3D-QSAR (CoMFA-Style) and Field Descriptors",
        "desc": "Conformer alignment, 3D interaction energy grids (steric Lennard-Jones and electrostatic Coulomb probes), and Partial Least Squares (PLS) regression.",
        "topics": ["CoMFA", "Conformer Align", "3D Grid Fields", "PLS Regression"],
        "filename": "nb14_3d_qsar_comfa_fields.ipynb"
    },
    {
        "id": "nb15",
        "part_label": "Part E · QSAR & ML",
        "cat": "qsar_ml",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~90 min",
        "title": "Drug-Likeness Filters and In-Silico ADME/T",
        "desc": "Lipinski Rule of 5, Veber oral bioavailability, Ghose, PAINS substructure filtering, Brenk structural alerts, Synthetic Accessibility (SA) score.",
        "topics": ["Lipinski Rule of 5", "Veber Rules", "PAINS Filter", "SA Score"],
        "filename": "nb15_druglikeness_and_admet.ipynb"
    },
    {
        "id": "nb16",
        "part_label": "Part F · Structure-Based",
        "cat": "structure_based",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~120 min",
        "title": "Target Discovery and Structure Preparation",
        "desc": "Differential gene expression volcano plots, subtractive genomics, PDB inspection, pocket detection, protonation of His/Asp/Glu, and receptor PDBQT prep.",
        "topics": ["Target Discovery", "Differential Expr", "PDB Prep", "Protonation"],
        "filename": "nb16_target_discovery_and_prep.ipynb"
    },
    {
        "id": "nb17",
        "part_label": "Part F · Structure-Based",
        "cat": "structure_based",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~120 min",
        "title": "Scoring Functions and Conformational Sampling",
        "desc": "Physics-based, empirical, and knowledge-based scoring functions. AutoDock energy decomposition. Metropolis Monte Carlo simulated annealing in toy potentials.",
        "topics": ["Scoring Functions", "Energy Decomp", "Monte Carlo", "Simulated Annealing"],
        "filename": "nb17_scoring_functions_and_sampling.ipynb"
    },
    {
        "id": "nb18",
        "part_label": "Part F · Structure-Based",
        "cat": "structure_based",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · ~150 min",
        "title": "AutoDock Vina Docking and Virtual Screening",
        "desc": "End-to-end Vina docking pipeline, Meeko ligand preparation, redocking validation (RMSD < 2.0 Å), experimental pIC50 correlation, and EF1%/BEDROC evaluation.",
        "topics": ["AutoDock Vina", "Meeko PDBQT", "Redocking RMSD", "EF1% & BEDROC"],
        "filename": "nb18_docking_and_virtual_screening.ipynb"
    },
    {
        "id": "nb19",
        "part_label": "Part F · Structure-Based",
        "cat": "structure_based",
        "diff_level": "Intermediate",
        "diff_class": "bg-amber-50 text-amber-700 border-amber-200",
        "dots_time": "●●○ · ~120 min",
        "title": "De Novo Design and Fragment-Based Discovery (FBDD)",
        "desc": "Pocket feature mapping, Congreve's Rule of Three, Ligand Efficiency (LE) and Lipophilic Efficiency (LLE), BRICS fragment growing/linking, and SA scoring.",
        "topics": ["Congreve Rule of 3", "Ligand Efficiency", "BRICS Fragments", "SA Scoring"],
        "filename": "nb19_denovo_design_and_fbdd.ipynb"
    },
    {
        "id": "nb20",
        "part_label": "Part G · Capstone",
        "cat": "capstone",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · Capstone Template",
        "title": "Capstone Project Template: Ligand-Based ML Model",
        "desc": "Fully parameterized master template for student projects: data curation, EDA, multi-representation featurization, scaffold splitting, modeling, and external screening.",
        "topics": ["Capstone Template", "Multi-Featurization", "Scaffold Split", "Prospective Screen"],
        "filename": "nb20_project_template_ligand_model.ipynb"
    },
    {
        "id": "nb21",
        "part_label": "Part G · Capstone",
        "cat": "capstone",
        "diff_level": "Advanced",
        "diff_class": "bg-purple-50 text-purple-700 border-purple-200",
        "dots_time": "●●● · Advanced · ~120 min",
        "title": "Graph Neural Networks for Property Prediction",
        "desc": "Molecular graph featurization, message-passing layers, Graph Convolutional Networks (GCN), global pooling readout, and ESOL aqueous solubility regression.",
        "topics": ["Graph Neural Nets", "Message Passing", "GCN Layers", "ESOL Solubility"],
        "filename": "nb21_gnn_property_prediction.ipynb"
    }
]

# Generate filter buttons
filter_buttons_html = """                <!-- Filter Tabs -->
                <div class="flex flex-wrap gap-2 text-xs font-semibold">
                    <button class="filter-btn active px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="all" onclick="filterNotebooks('all', this)">All (22)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="foundations" onclick="filterNotebooks('foundations', this)">Foundations (6)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="representations" onclick="filterNotebooks('representations', this)">Representations (2)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="descriptors" onclick="filterNotebooks('descriptors', this)">Descriptors (2)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="searching" onclick="filterNotebooks('searching', this)">Searching (2)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="qsar_ml" onclick="filterNotebooks('qsar_ml', this)">QSAR & ML (4)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="structure_based" onclick="filterNotebooks('structure_based', this)">Structure-Based (4)</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg border border-slate-200 text-slate-700 hover:border-teal-400 transition" data-filter="capstone" onclick="filterNotebooks('capstone', this)">Capstone (2)</button>
                </div>"""

cards_html_list = []
for d in nb_data:
    topics_spans = "\n".join([f'                            <span class="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-[11px] font-medium">{t}</span>' for t in d["topics"]])
    card = f"""                <!-- {d["id"]} -->
                <div class="nb-card {d["cat"]} bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition space-y-4 flex flex-col justify-between" data-category="{d["cat"]}">
                    <div class="space-y-3">
                        <div class="flex items-center justify-between text-xs">
                            <span class="font-bold text-teal-600 uppercase tracking-wide">{d["part_label"]} · {d["id"]}</span>
                            <div class="flex items-center space-x-2">
                                <span class="px-2 py-0.5 rounded-full text-[11px] font-semibold {d["diff_class"]} border">{d["diff_level"]}</span>
                                <span class="text-slate-400 font-medium">{d["dots_time"]}</span>
                            </div>
                        </div>
                        <h3 class="text-lg font-bold text-slate-900 leading-snug">{d["title"]}</h3>
                        <p class="text-sm text-slate-600 leading-relaxed">{d["desc"]}</p>
                        <div class="flex flex-wrap gap-1.5 pt-1">
{topics_spans}
                        </div>
                    </div>
                    <div class="flex items-center justify-between pt-3 border-t border-slate-100 text-xs">
                        <span class="code-pill text-slate-500 truncate max-w-[170px]" title="{d["filename"]}">{d["filename"]}</span>
                        <div class="flex items-center space-x-2">
                            <a href="https://github.com/meprateekpaul/Cheminformatics/blob/main/notebooks/{d["filename"]}" target="_blank" class="inline-flex items-center space-x-1 text-slate-600 hover:text-teal-600 font-semibold px-2 py-1 rounded hover:bg-slate-50 transition" title="View on GitHub">
                                <i class="fa-brands fa-github text-sm"></i>
                                <span>GitHub</span>
                            </a>
                            <a href="https://colab.research.google.com/github/meprateekpaul/Cheminformatics/blob/main/notebooks/{d["filename"]}" target="_blank" class="inline-flex items-center space-x-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-800 px-2.5 py-1 rounded border border-amber-300 font-semibold transition" title="Open in Google Colab">
                                <i class="fa-solid fa-play text-[10px] text-amber-600"></i>
                                <span>Colab</span>
                            </a>
                        </div>
                    </div>
                </div>"""
    cards_html_list.append(card)

cards_grid_content = "\n\n".join(cards_html_list)

filter_script = """    <!-- Filter Script -->
    <script>
        function filterNotebooks(category, btn) {
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(b => b.classList.remove('active'));
            if (btn) {
                btn.classList.add('active');
            } else {
                const targetBtn = document.querySelector(`.filter-btn[data-filter="${category}"]`);
                if (targetBtn) targetBtn.classList.add('active');
            }

            const cards = document.querySelectorAll('.nb-card');
            cards.forEach(card => {
                const cardCat = card.getAttribute('data-category');
                if (category === 'all' || cardCat === category || card.classList.contains(category)) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>"""

files_to_update = [
    "/Users/prateekpaul/Library/CloudStorage/GoogleDrive-prateekp@iiitd.ac.in/My Drive/IIITD/mygithub/meprateekpaul.github.io/cheminformatics.html",
    "/Users/prateekpaul/Library/CloudStorage/GoogleDrive-prateekp@iiitd.ac.in/My Drive/IIITD/mygithub/Cheminformatics/index.html"
]

for file_path in files_to_update:
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Add id="about" to hero section if not present
    if '<section id="about"' not in html:
        html = re.sub(
            r'<section class="hero-mesh',
            r'<section id="about" class="hero-mesh',
            html
        )

    # 2. Replace Filter Tabs & Cards Grid
    pattern = r'<!-- Filter Tabs -->\s*<div class="flex flex-wrap gap-2 text-xs font-semibold">.*?</div>\s*</div>\s*<!-- Notebook Cards Grid -->\s*<div id="notebook-grid" class="grid grid-cols-1 md:grid-cols-2 gap-6">.*?</div>\s*</section>\s*<!-- Datasets Registry Table -->'
    replacement = filter_buttons_html + "\n            </div>\n\n            <!-- Notebook Cards Grid -->\n            <div id=\"notebook-grid\" class=\"grid grid-cols-1 md:grid-cols-2 gap-6\">\n\n" + cards_grid_content + "\n            </div>\n        </section>\n\n        <!-- Datasets Registry Table -->"

    html, count = re.subn(pattern, replacement, html, flags=re.DOTALL)
    print(f"Replaced cards in {file_path}: {count} occurrences")

    # 3. Replace script
    script_pattern = r'<!-- Filter Script -->\s*<script>.*?</script>\s*</body>'
    html, count_s = re.subn(script_pattern, filter_script + "\n</body>", html, flags=re.DOTALL)
    print(f"Replaced script in {file_path}: {count_s} occurrences")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Successfully wrote updated file:", file_path)
