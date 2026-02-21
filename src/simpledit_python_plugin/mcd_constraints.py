"""
MCD Constraint Generator

Algorithmically generates MCD constraints by comparing reactant and product
bond graphs.  No LLM required — the bond changes are read directly from the
explicit connectivity stored in atoms.info["connectivity"] (populated by
atoms_from_sdf from the SDF bond table).

Detected change types
---------------------
- Bond breaking  : bond in reactant, absent in product  → stretch to ``breaking_distance``
- Bond forming   : bond in product, absent in reactant  → contract to typical equilibrium length
- Order change   : same atom pair, different bond order  → adjust to typical equilibrium length

Step count heuristic: 15 steps per Å of coordinate change, clamped to [10, 40].
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Typical equilibrium bond lengths (Å) by element pair + bond order
# ---------------------------------------------------------------------------

_BOND_LENGTHS: Dict[FrozenSet, Dict[float, float]] = {
    frozenset({"C", "C"}):  {1.0: 1.54, 1.5: 1.40, 2.0: 1.34, 3.0: 1.20},
    frozenset({"C", "N"}):  {1.0: 1.47, 1.5: 1.34, 2.0: 1.27, 3.0: 1.15},
    frozenset({"C", "O"}):  {1.0: 1.43, 1.5: 1.36, 2.0: 1.23},
    frozenset({"C", "S"}):  {1.0: 1.82, 2.0: 1.61},
    frozenset({"C", "H"}):  {1.0: 1.09},
    frozenset({"C", "F"}):  {1.0: 1.35},
    frozenset({"C", "Cl"}): {1.0: 1.76},
    frozenset({"C", "Br"}): {1.0: 1.94},
    frozenset({"C", "I"}):  {1.0: 2.14},
    frozenset({"C", "P"}):  {1.0: 1.84, 2.0: 1.67},
    frozenset({"N", "N"}):  {1.0: 1.45, 2.0: 1.25, 3.0: 1.10},
    frozenset({"N", "O"}):  {1.0: 1.45, 2.0: 1.22},
    frozenset({"N", "H"}):  {1.0: 1.01},
    frozenset({"O", "H"}):  {1.0: 0.96},
    frozenset({"O", "O"}):  {1.0: 1.48, 2.0: 1.21},
    frozenset({"S", "S"}):  {1.0: 2.05},
    frozenset({"S", "O"}):  {1.0: 1.65, 2.0: 1.43},
    frozenset({"S", "H"}):  {1.0: 1.34},
    frozenset({"P", "O"}):  {1.0: 1.63, 2.0: 1.48},
    frozenset({"P", "H"}):  {1.0: 1.42},
}

_BREAKING_DISTANCE = 3.5   # Å — beyond any normal bond
_STEPS_PER_ANGSTROM = 15
_MIN_STEPS = 10
_MAX_STEPS = 40


def _typical_length(elem1: str, elem2: str, order: float) -> float:
    """Return a typical equilibrium bond length for the given pair and order."""
    key = frozenset({elem1, elem2})
    table = _BOND_LENGTHS.get(key)
    if table is None:
        return 1.6  # generic fallback
    if order in table:
        return table[order]
    # Nearest bond order in table
    closest = min(table, key=lambda o: abs(o - order))
    return table[closest]


def _n_steps(current: float, target: float) -> int:
    delta = abs(target - current)
    return max(_MIN_STEPS, min(_MAX_STEPS, int(delta * _STEPS_PER_ANGSTROM)))


def generate_mcd_constraints(req) -> "MCDConstraintsResponse":
    """Compare reactant and product bond graphs and return MCD constraints.

    Parameters
    ----------
    req : MCDConstraintsRequest
        Has fields: reactant_sdf, product_sdf, n_steps (default per constraint),
        breaking_distance (Å target for bond breaking).

    Returns
    -------
    MCDConstraintsResponse
        Pydantic response with ``constraints`` list and ``description`` string.
    """
    # Import here to avoid circular import at module load time
    from .calculation import atoms_from_sdf
    from .calculation_api import MCDConstraintItem, MCDConstraintsResponse

    react = atoms_from_sdf(req.reactant_sdf)
    prod  = atoms_from_sdf(req.product_sdf)

    if "connectivity" not in react.info or "connectivity" not in prod.info:
        return MCDConstraintsResponse(
            constraints=[],
            description=(
                "Bond connectivity information is missing from one or both SDF inputs. "
                "Ensure the SDF files contain explicit bond tables (V2000/V3000 format)."
            ),
        )

    symbols   = react.get_chemical_symbols()
    positions = react.get_positions()

    def bond_dict(conn):
        """Return {(min_i, max_j): order} from connectivity list."""
        return {
            (min(int(b), int(e)), max(int(b), int(e))): float(o)
            for b, e, o in conn
        }

    react_bonds = bond_dict(react.info["connectivity"])
    prod_bonds  = bond_dict(prod.info["connectivity"])
    all_keys    = set(react_bonds) | set(prod_bonds)

    constraints: List[MCDConstraintItem] = []
    change_lines: List[str] = []

    for (i, j) in sorted(all_keys):
        r_order = react_bonds.get((i, j), 0.0)
        p_order = prod_bonds.get((i, j),  0.0)

        if r_order == p_order:
            continue

        ei, ej = symbols[i], symbols[j]
        current_dist = float(np.linalg.norm(positions[i] - positions[j]))

        if p_order == 0.0:
            # Bond breaking
            target = req.breaking_distance
            n = _n_steps(current_dist, target)
            constraints.append(
                MCDConstraintItem(indices=[i, j], target=round(target, 3), n_steps=n)
            )
            change_lines.append(
                f"BREAK  {ei}({i})-{ej}({j})  "
                f"order {r_order:.0f}→∅  "
                f"{current_dist:.2f}→{target:.1f} Å  "
                f"({n} steps)"
            )

        elif r_order == 0.0:
            # Bond forming
            target = _typical_length(ei, ej, p_order)
            n = _n_steps(current_dist, target)
            constraints.append(
                MCDConstraintItem(indices=[i, j], target=round(target, 3), n_steps=n)
            )
            change_lines.append(
                f"FORM   {ei}({i})-{ej}({j})  "
                f"order ∅→{p_order:.0f}  "
                f"{current_dist:.2f}→{target:.2f} Å  "
                f"({n} steps)"
            )

        else:
            # Bond order change (e.g. C=C → C-C in a [4+2])
            target = _typical_length(ei, ej, p_order)
            n = _n_steps(current_dist, target)
            constraints.append(
                MCDConstraintItem(indices=[i, j], target=round(target, 3), n_steps=n)
            )
            change_lines.append(
                f"CHANGE {ei}({i})-{ej}({j})  "
                f"order {r_order:.0f}→{p_order:.0f}  "
                f"{current_dist:.2f}→{target:.2f} Å  "
                f"({n} steps)"
            )

    if change_lines:
        description = (
            f"Detected {len(constraints)} bond change(s):\n"
            + "\n".join(f"  • {line}" for line in change_lines)
        )
    else:
        description = "No bond changes detected between reactant and product."

    return MCDConstraintsResponse(constraints=constraints, description=description)
