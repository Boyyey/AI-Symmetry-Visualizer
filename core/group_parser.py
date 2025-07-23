from sympy.combinatorics.named_groups import DihedralGroup, CyclicGroup, AlternatingGroup, SymmetricGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics import Permutation
import networkx as nx

# Add more imports as needed

def get_cayley_table(G):
    try:
        elements = list(G.generate(af=True))
        table = []
        for a in elements:
            row = []
            for b in elements:
                prod = a * b
                idx = elements.index(prod)
                row.append(idx)
            table.append(row)
        return elements, table
    except Exception as e:
        return [], []

def get_cayley_graph(G):
    try:
        elements = list(G.generate(af=True))
        generators = G.generators
        edges = []
        for g in generators:
            for e in elements:
                target = g * e
                edges.append((elements.index(e), elements.index(target), str(g)))
        return elements, edges
    except Exception as e:
        return [], []

def parse_group(group_input: str) -> dict:
    group_input = group_input.strip().upper()
    info = {"input": group_input, "order": None, "type": None, "generators": [], "cayley_table": None, "cayley_elements": None, "cayley_edges": None, "error": None}
    try:
        if group_input.startswith('D') and group_input[1:].isdigit():
            n = int(group_input[1:])
            G = DihedralGroup(n)
        elif group_input.startswith('C') and group_input[1:].isdigit():
            n = int(group_input[1:])
            G = CyclicGroup(n)
        elif group_input.startswith('A') and group_input[1:].isdigit():
            n = int(group_input[1:])
            G = AlternatingGroup(n)
        elif group_input.startswith('S') and group_input[1:].isdigit():
            n = int(group_input[1:])
            G = SymmetricGroup(n)
        else:
            info["error"] = "Unrecognized group format. Try D4, C6, A5, S3, etc."
            return info
        info["order"] = G.order()
        info["type"] = str(G)
        info["generators"] = [str(g) for g in G.generators]
        elements, table = get_cayley_table(G)
        info["cayley_elements"] = [str(e) for e in elements]
        info["cayley_table"] = table
        _, edges = get_cayley_graph(G)
        info["cayley_edges"] = edges
    except Exception as e:
        info["error"] = f"Group parsing error: {e}"
    return info 