def generate_explanation(group_info, style):
    if group_info.get('error'):
        return "Cannot generate explanation due to group parsing error."
    return f"This {style.lower()} visualization uses the {group_info.get('type', 'unknown group')} of order {group_info.get('order', '?')}. Symmetry operations are applied to create mesmerizing patterns. (More details coming soon!)"

def suggest_next_group(group_info):
    t = group_info.get('type', '')
    order = group_info.get('order', None)
    if 'Dihedral' in t and order:
        n = int(order // 2)
        return f"Try the cyclic group C{n} for pure rotations, or D{n+1} for more complexity!"
    if 'Cyclic' in t and order:
        return f"Try the dihedral group D{order} to add reflections, or C{order+1} for a higher order!"
    if 'Alternating' in t:
        return "Try S4 (octahedral symmetry) or A5 (icosahedral symmetry) for beautiful 3D visuals!"
    if 'Symmetric' in t and order:
        return f"Try A{order} (alternating group) or S{order+1} for more permutations!"
    return "Try a different group type or increase the order for new symmetries!" 