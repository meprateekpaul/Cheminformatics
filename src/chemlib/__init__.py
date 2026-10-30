"""chemlib - A modern, pedagogical cheminformatics library supporting lecture-aligned workflows.
"""

__version__ = "0.1.0"

from chemlib import io
from chemlib import standardize
from chemlib import viz
from chemlib import graphs
from chemlib import descriptors
from chemlib import fingerprints
from chemlib import metrics
from chemlib import pharm
from chemlib import docking

__all__ = [
    "io",
    "standardize",
    "viz",
    "graphs",
    "descriptors",
    "fingerprints",
    "metrics",
    "pharm",
    "docking",
]
