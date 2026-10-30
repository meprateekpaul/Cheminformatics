# Changelog

All notable changes to the Cheminformatics course repository and `chemlib` library are documented here.

## [0.1.0] - 2026-10-30
### Added
- **Dedicated Conda Environment (`cheminf`):** Built with Python 3.11, RDKit 2026, Open Babel, AutoDock Vina, Meeko, MordredCommunity, Scikit-learn, BioPython, and Py3Dmol.
- **`chemlib` Package:** Reusable modular architecture covering `io`, `standardize`, `viz`, `graphs`, `descriptors`, `fingerprints`, `metrics`, `pharm`, and `docking`.
- **Unit Test Suite (`tests/`):** Comprehensive unit tests verifying matrices, topological indices, descriptors, fingerprints, metrics, and pharmacophores.
- **Curated Dataset Registry (`data/`):** Automated fetcher (`data/fetch_data.py`) generating D01 through D18 with documented provenance in `data/README.md`.
- **Course Infrastructure:** Root `README.md`, `LICENSE`, `Makefile`, `environment.yml`, `requirements.txt`, and GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- **Graded Assignments & Capstone Project:** Briefs, rubrics, and templates for Assignment A1 (Library Search & Pharmacophores), Assignment A2 (De Novo Design & FBDD), and the Capstone Project.
- **Documentation:** Modernized lecture slide index with errata callouts (`docs/slides_index.md`), course glossary (`docs/glossary.md`), and comprehensive question bank (`docs/question_bank.md`).
