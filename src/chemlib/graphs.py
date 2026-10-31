"""chemlib.graphs - Graph theory representations, matrices, topological indices, and isomorphism."""

from __future__ import annotations

import networkx as nx
import numpy as np
from rdkit import Chem


def mol_to_networkx(mol: Chem.Mol, explicit_hydrogens: bool = False) -> nx.Graph:
    """Convert an RDKit Mol to a networkx Graph with atom and bond attributes."""
    if explicit_hydrogens:
        working_mol = Chem.AddHs(mol)
    else:
        working_mol = Chem.RemoveHs(mol)

    G = nx.Graph()
    for atom in working_mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            element=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            is_aromatic=atom.GetIsAromatic(),
            degree=atom.GetDegree(),
            total_valence=atom.GetTotalValence(),
        )

    for bond in working_mol.GetBonds():
        b_type = bond.GetBondTypeAsDouble()
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),
            bond_order=b_type,
            is_aromatic=bond.GetIsAromatic(),
        )

    return G


def get_adjacency_matrix(mol: Chem.Mol) -> np.ndarray:
    """Return the unweighted 0/1 adjacency matrix of heavy atoms."""
    return Chem.GetAdjacencyMatrix(mol)


def get_bond_order_matrix(mol: Chem.Mol) -> np.ndarray:
    """Return the bond-order weighted adjacency matrix."""
    n = mol.GetNumAtoms()
    mat = np.zeros((n, n), dtype=float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bo = bond.GetBondTypeAsDouble()
        mat[i, j] = bo
        mat[j, i] = bo
    return mat


def get_distance_matrix(mol: Chem.Mol) -> np.ndarray:
    """Return topological distance matrix (shortest path between all heavy atom pairs)."""
    return Chem.GetDistanceMatrix(mol)


def molecular_walk_count(mol: Chem.Mol, k: int) -> int:
    """Compute molecular walk count of length k: MWC_k = sum of all entries of A^k.
    MWC_0 = number of atoms
    MWC_1 = 2 * number of bonds
    """
    if k == 0:
        return mol.GetNumHeavyAtoms()
    A = get_adjacency_matrix(mol)
    A_k = np.linalg.matrix_power(A, k)
    return int(np.sum(A_k))


def self_returning_walk_count(mol: Chem.Mol, k: int) -> int:
    """Compute self-returning walk count of length k: SRW_k = Trace(A^k)."""
    A = get_adjacency_matrix(mol)
    A_k = np.linalg.matrix_power(A, k)
    return int(np.trace(A_k))


def wiener_index(mol: Chem.Mol) -> int:
    """Compute the Wiener Index W: half sum of all shortest topological path distances.
    W = 0.5 * sum_{i,j} D_ij
    """
    D = get_distance_matrix(mol)
    return int(np.sum(D) // 2)


def zagreb_m1_index(mol: Chem.Mol) -> int:
    """Compute first Zagreb index M1 = sum_i (delta_i)^2 over all vertices."""
    return sum(atom.GetDegree() ** 2 for atom in mol.GetAtoms())


def zagreb_m2_index(mol: Chem.Mol) -> int:
    """Compute second Zagreb index M2 = sum_{edges (i,j)} delta_i * delta_j."""
    return sum(
        bond.GetBeginAtom().GetDegree() * bond.GetEndAtom().GetDegree() for bond in mol.GetBonds()
    )


def randic_index(mol: Chem.Mol) -> float:
    """Compute the Randić branching/connectivity index chi:
    chi = sum_{edges (i,j)} 1 / sqrt(delta_i * delta_j)
    """
    val = 0.0
    for bond in mol.GetBonds():
        d1 = bond.GetBeginAtom().GetDegree()
        d2 = bond.GetEndAtom().GetDegree()
        if d1 > 0 and d2 > 0:
            val += 1.0 / np.sqrt(d1 * d2)
    return float(val)


def check_subgraph_isomorphism_vf2(
    target_mol: Chem.Mol,
    query_mol: Chem.Mol,
    match_elements: bool = True,
    match_bonds: bool = True,
) -> bool:
    """Check if query_mol is a subgraph of target_mol using NetworkX VF2 algorithm."""
    G_target = mol_to_networkx(target_mol)
    G_query = mol_to_networkx(query_mol)

    node_match = None
    edge_match = None

    if match_elements:

        def node_match(n1, n2):
            return n1["element"] == n2["element"]

    if match_bonds:

        def edge_match(e1, e2):
            return e1["bond_order"] == e2["bond_order"]

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        G_target, G_query, node_match=node_match, edge_match=edge_match
    )
    return matcher.subgraph_is_isomorphic()
