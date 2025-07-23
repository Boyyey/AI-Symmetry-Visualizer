from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.named_groups import DihedralGroup, CyclicGroup, AlternatingGroup, SymmetricGroup
from sympy.combinatorics import Permutation
import networkx as nx

# --- Subgroup lattice ---
def subgroup_lattice(G):
    """Return a networkx DiGraph representing the subgroup lattice of G."""
    # For small groups only (sympy limitation)
    subgroups = list(G.subgroups())
    lattice = nx.DiGraph()
    for H in subgroups:
        lattice.add_node(str(H))
    for H in subgroups:
        for K in subgroups:
            if H.order() < K.order() and H.is_subgroup(K):
                lattice.add_edge(str(H), str(K))
    return lattice

# --- Element order distribution ---
def element_orders(G):
    """Return a dict mapping order -> count of elements of that order."""
    orders = {}
    for g in G.generate(af=True):
        o = g.order()
        orders[o] = orders.get(o, 0) + 1
    return orders

# --- Conjugacy classes ---
def conjugacy_classes(G):
    """Return a list of sets, each set is a conjugacy class."""
    return [set(map(str, c)) for c in G.conjugacy_classes()]

# --- Character table (for small groups) ---
def character_table(G):
    """Return a character table as a list of lists (rows: irreps, cols: classes)."""
    try:
        return G.character_table()
    except Exception:
        return None 