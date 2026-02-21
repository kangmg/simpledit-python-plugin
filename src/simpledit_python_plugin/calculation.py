"""
Calculation module for ASE + MLIP based computational chemistry

Architecture
------------
Core functions take ASE Atoms directly and return typed dataclasses.
Serialization (Atoms <-> SDF) lives in a separate section and is only
used at the API boundary (FastAPI endpoints in main.py).

    Local use:  atoms → core_fn(atoms, attach_fn) → Result
    API use:    sdf → atoms_from_sdf → core_fn → Result → atoms_to_sdf → sdf

SDF is the canonical I/O format because it preserves:
  - Explicit bond orders and bond types (simpledit works with explicit bonds)
  - 3D coordinates
  - Fragment/disconnected-component information

Bond preservation via atoms.info["connectivity"]
-------------------------------------------------
ASE Atoms objects carry only element symbols + positions; bond topology is
not stored internally by ASE.  To bridge this gap, ``atoms_from_sdf``
embeds the bond graph into ``atoms.info["connectivity"]`` as a list of
``(begin_idx, end_idx, bond_order_float)`` tuples.

Because ``atoms.info`` is preserved through ``atoms.copy()`` and most
ASE operations, the bond info automatically travels with the structure
through optimization / NEB / IRC without requiring a separate RDKit
object to be kept in scope.

``atoms_to_sdf`` reads back from ``atoms.info["connectivity"]`` to
reconstruct the RDKit Mol with the original topology before serializing —
never infers bonds from geometry.

    atoms  = atoms_from_sdf(sdf)          # bond info in atoms.info
    result = geometry_optimize(atoms, …)  # info preserved through copy
    sdf    = atoms_to_sdf(result.atoms)   # exact original bonds restored

Reaction modeling — reactant → product via MCD
-----------------------------------------------
Embedding reactant and product separately from SMILES is problematic for
reaction modeling: random chain conformations introduce noise that
interferes with NEB/IRC initial paths.

The correct approach is:
  1. Build the reactant structure (from SDF, geometry optimization, etc.)
  2. Deform that structure into the product geometry via ``run_mcd``

``run_mcd`` uses Multi-Coordinate Driving (asemcd2) to incrementally
drive selected bond distances, angles, or dihedrals toward target values
while relaxing all other degrees of freedom at each step.  The resulting
endpoint is a product structure that shares the same conformational
scaffold as the reactant — ideal as an NEB/DMF endpoint.

Supported calculations
----------------------
- Geometry optimization  : geometry_optimize  (BFGS / LBFGS / FIRE)
- TS optimization        : ts_optimize        (Sella)
- Double-ended search    : run_neb            (NEB + IDPP)
                           run_dmf            (DirectMaxFlux)
                           run_popcornn       (NN continuous path, MLIP)
- IRC                    : run_irc            (Sella IRC, forward + reverse)
- Vibrational analysis   : freq_analysis      (ASE Vibrations)
- MCD deformation        : run_mcd            (asemcd2, reactant → product)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional, Union

import ase
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read as ase_read, write as ase_write
from ase.optimize import BFGS, FIRE, LBFGS

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A callable that attaches a calculator to one Atoms or a list of Atoms in-place
AttachFn = Callable[[Union[Atoms, List[Atoms]]], None]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of geometry optimization."""
    atoms: Atoms
    converged: bool
    steps: int


@dataclass
class TSOptimizationResult:
    """Result of transition state optimization (Sella)."""
    atoms: Atoms
    converged: bool
    steps: int


@dataclass
class NEBResult:
    """Result of NEB calculation."""
    images: List[Atoms]
    converged: bool
    n_images: int
    method: str


@dataclass
class DMFResult:
    """Result of DirectMaxFlux calculation."""
    images: List[Atoms]
    nmove: int
    convergence: str


@dataclass
class IRCResult:
    """Result of IRC calculation.

    ``path`` is ordered as: reactant side → TS → product side.
    """
    path: List[Atoms]
    forward_converged: bool
    reverse_converged: bool


@dataclass
class FreqResult:
    """Result of vibrational frequency analysis."""
    frequencies: List[float]   # cm⁻¹; imaginary freqs are negative
    n_imaginary: int
    hessian: np.ndarray        # (3N, 3N) mass-weighted Hessian


@dataclass
class PopcornnResult:
    """Result of a Popcornn neural-network path optimization.

    Uses a continuous NN path representation with MLIP potentials to find
    optimal reaction pathways and transition states.  Multi-stage
    optimization is supported: e.g. geodesic interpolation followed by
    MLIP-based variational refinement.
    """
    images: List[Atoms]
    ts_atoms: Atoms
    energies: np.ndarray
    forward_barrier: float   # eV, relative to images[0]
    reverse_barrier: float   # eV, relative to images[-1]


@dataclass
class MCDResult:
    """Result of a Multi-Coordinate Driving scan.

    The pathway is ordered from the starting (reactant) geometry to the
    endpoint reached after all constraints have been satisfied.  Bond
    connectivity from the input Atoms is propagated to every frame so that
    the full trajectory can be exported as a bond-preserving SDF.
    """
    pathway: List[Atoms]
    energies: np.ndarray
    ts_index: int       # index of the highest-energy frame in pathway
    ts_atoms: Atoms     # frame at ts_index (TS guess)
    forward_barrier: float   # eV, relative to pathway[0]
    reverse_barrier: float   # eV, relative to pathway[-1]


# ---------------------------------------------------------------------------
# Calculator helpers
# ---------------------------------------------------------------------------

def make_attach_fn(calc_getter: Callable[[], Calculator]) -> AttachFn:
    """Wrap a no-arg calculator factory into an AttachFn.

    Bridges the existing calculators/ interface (which returns a Calculator
    object) with the AttachFn pattern used by core calculation functions.

    Example
    -------
    >>> from simpledit_python_plugin.calculators.xtb import get_xtb_calculator
    >>> attach = make_attach_fn(lambda: get_xtb_calculator(charge=0, spin=1))
    >>> geometry_optimize(atoms, attach=attach)
    """
    def _attach(images: Union[Atoms, List[Atoms]]) -> None:
        if isinstance(images, Atoms):
            images.calc = calc_getter()
        else:
            for atoms in images:
                atoms.calc = calc_getter()
    return _attach


# ---------------------------------------------------------------------------
# Serialization utilities  (used only at the API boundary)
#
# SDF is the canonical format: bond orders, bond types, and 3D coordinates
# are all preserved.  Bond info is embedded into atoms.info["connectivity"]
# so it travels with the Atoms object through ASE computations — no need to
# carry a separate RDKit Mol object around.
#
# atoms.info["connectivity"] layout
# ----------------------------------
#   list of (begin_atom_idx, end_atom_idx, bond_order_float)
#   bond_order_float: 1.0 = SINGLE, 1.5 = AROMATIC, 2.0 = DOUBLE, 3.0 = TRIPLE
#
# Recommended round-trip
# ----------------------
#   atoms  = atoms_from_sdf(sdf)          # bond info embedded in atoms.info
#   result = geometry_optimize(atoms, …)  # atoms.info survives .copy() / ase ops
#   sdf    = atoms_to_sdf(result.atoms)   # bonds reconstructed from atoms.info
#
# For multi-fragment SDF (reactant + product in one file, salts, etc.)
#   blocks = split_sdf(sdf)               # split on "$$$$"
#   frames = [atoms_from_sdf(b) for b in blocks]
# ---------------------------------------------------------------------------

# Bond-order float → RDKit BondType mapping used in both directions
_BOND_ORDER_TO_TYPE: dict = {}   # populated lazily to avoid rdkit import at module load
_BOND_TYPE_TO_ORDER: dict = {}


def _bond_type_maps():
    """Return (order→type, type→order) dicts, importing rdkit once."""
    if not _BOND_ORDER_TO_TYPE:
        from rdkit.Chem import BondType
        _BOND_ORDER_TO_TYPE.update({
            1.0: BondType.SINGLE,
            1.5: BondType.AROMATIC,
            2.0: BondType.DOUBLE,
            3.0: BondType.TRIPLE,
        })
        _BOND_TYPE_TO_ORDER.update({v: k for k, v in _BOND_ORDER_TO_TYPE.items()})
    return _BOND_ORDER_TO_TYPE, _BOND_TYPE_TO_ORDER


def atoms_from_sdf(sdf: str) -> Atoms:
    """Parse a single SDF block into ASE Atoms.

    Bond topology is embedded into ``atoms.info["connectivity"]`` as a list of
    ``(begin_idx, end_idx, bond_order_float)`` tuples so it is preserved
    through ASE copy/computation without requiring a separate RDKit object.
    Formal charges are stored via ``atoms.set_initial_charges`` when non-zero.
    """
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(sdf, removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("Failed to parse SDF block")

    _, type_to_order = _bond_type_maps()

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    atoms = Atoms(symbols=symbols, positions=positions)

    # Embed bond topology — survives atoms.copy() and slicing
    atoms.info["connectivity"] = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
         type_to_order.get(b.GetBondType(), 1.0))
        for b in mol.GetBonds()
    ]

    # Preserve formal charges when present
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    if any(c != 0 for c in charges):
        atoms.set_initial_charges(charges)

    return atoms


def _atoms_to_mol(atoms: Atoms):
    """Rebuild an RDKit Mol from ASE Atoms using stored connectivity.

    Reads ``atoms.info["connectivity"]`` to reconstruct bond topology without
    any geometry-based bond inference.  Raises ``KeyError`` if connectivity
    has not been stored (i.e., the Atoms object did not originate from
    ``atoms_from_sdf``).
    """
    from rdkit import Chem

    order_to_type, _ = _bond_type_maps()
    connectivity = atoms.info["connectivity"]  # raises KeyError if absent

    rw = Chem.RWMol()
    for symbol in atoms.get_chemical_symbols():
        rw.AddAtom(Chem.Atom(symbol))

    # Restore formal charges if stored
    try:
        charges = atoms.get_initial_charges()
        for i, c in enumerate(charges):
            if c != 0:
                rw.GetAtomWithIdx(i).SetFormalCharge(int(round(c)))
    except Exception:
        pass

    for begin, end, order in connectivity:
        bond_type = order_to_type.get(float(order), Chem.BondType.SINGLE)
        rw.AddBond(int(begin), int(end), bond_type)

    conf = Chem.Conformer(len(atoms))
    for i, pos in enumerate(atoms.positions):
        conf.SetAtomPosition(i, pos.tolist())
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def atoms_to_sdf(atoms: Atoms) -> str:
    """Serialize ASE Atoms back to an SDF block.

    Bond topology is read from ``atoms.info["connectivity"]`` (embedded by
    ``atoms_from_sdf``), so the original bond orders are preserved even after
    geometry optimization, NEB, or IRC.

    Falls back to geometry-based bond inference (``DetermineBonds``) with a
    warning when connectivity is not stored — this is lossy and should be
    avoided for structures that originally came from SDF input.
    """
    from rdkit import Chem

    if "connectivity" in atoms.info:
        mol = _atoms_to_mol(atoms)
        return Chem.MolToMolBlock(mol)

    # Fallback: geometry-based bond inference (lossy — warns the caller)
    import warnings
    from rdkit.Chem import AllChem
    from io import StringIO as _StringIO
    warnings.warn(
        "atoms_to_sdf: atoms.info['connectivity'] not found — falling back to "
        "DetermineBonds, which may produce incorrect bond orders.  Prefer "
        "atoms_from_sdf to parse input so bond info is preserved.",
        stacklevel=2,
    )
    buf = _StringIO()
    ase_write(buf, atoms, format="xyz")
    mol = Chem.MolFromXYZBlock(buf.getvalue())
    if mol is None:
        raise ValueError("Failed to convert Atoms to RDKit Mol via XYZ fallback")
    AllChem.DetermineBonds(mol)
    return Chem.MolToMolBlock(mol)


# -- Multi-fragment helpers --------------------------------------------------

def split_sdf(sdf: str) -> List[str]:
    """Split a multi-molecule SDF into a list of individual SDF blocks.

    SDF files use ``$$$$`` as a record separator.  Each returned string is a
    valid standalone SDF block with the terminator line included.

    Useful for reactant/product pairs delivered as a single file.
    """
    blocks: List[str] = []
    current: List[str] = []
    for line in sdf.splitlines(keepends=True):
        current.append(line)
        if line.strip() == "$$$$":
            block = "".join(current).strip()
            if block:
                blocks.append(block)
            current = []
    remainder = "".join(current).strip()
    if remainder:
        blocks.append(remainder)
    return blocks


def atoms_list_from_sdf(sdf: str) -> List[Atoms]:
    """Parse a multi-molecule SDF into a list of ASE Atoms objects.

    Each Atoms object carries bond info in ``atoms.info["connectivity"]``.
    Equivalent to ``[atoms_from_sdf(b) for b in split_sdf(sdf)]``.
    """
    return [atoms_from_sdf(block) for block in split_sdf(sdf)]


def images_to_sdf(images: List[Atoms]) -> str:
    """Serialize a trajectory / NEB / MCD path to a multi-molecule SDF.

    Each frame becomes one SDF record joined with ``$$$$`` separators.
    Bond topology is read from each frame's ``atoms.info["connectivity"]``,
    which is propagated automatically by ``run_mcd`` / ``run_neb`` etc.
    """
    blocks = [atoms_to_sdf(img) for img in images]
    return "\n$$$$\n".join(blocks) + "\n$$$$\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_calc(atoms: Atoms, attach: Optional[AttachFn]) -> None:
    """Attach a calculator if the atoms object has none."""
    if atoms.calc is None:
        if attach is None:
            raise ValueError(
                "Atoms object has no calculator and no attach_fn was provided."
            )
        attach(atoms)


def _pre_optimize_endpoints(
    reactant: Atoms,
    product: Atoms,
    attach: AttachFn,
    optimizer=FIRE,
    fmax: float = 0.05,
    steps: int = 200,
) -> None:
    """Optimize reactant and product before a double-ended search."""
    attach([reactant, product])
    for mol in (reactant, product):
        opt = optimizer(mol)
        opt.run(fmax=fmax, steps=steps)


# ---------------------------------------------------------------------------
# Core calculation functions
# ---------------------------------------------------------------------------

def geometry_optimize(
    atoms: Atoms,
    attach: Optional[AttachFn] = None,
    optimizer=BFGS,
    fmax: float = 0.05,
    steps: int = 200,
    logfile: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> OptimizationResult:
    """Geometry optimization.

    Parameters
    ----------
    atoms:
        Structure to optimize (modified in-place).
    attach:
        AttachFn to set the calculator. Used only if ``atoms.calc`` is None.
    optimizer:
        ASE optimizer class (BFGS, LBFGS, FIRE, …).
    fmax:
        Force convergence criterion (eV/Å).
    steps:
        Maximum number of steps.
    logfile:
        Path for the optimizer log. ``None`` suppresses output.
    trajectory:
        Path for the .traj file. ``None`` disables trajectory writing.
    """
    _ensure_calc(atoms, attach)
    opt = optimizer(atoms, logfile=logfile, trajectory=trajectory)
    converged = opt.run(fmax=fmax, steps=steps)
    if not converged:
        print("[geometry_optimize] WARNING: did not converge")
    return OptimizationResult(atoms=atoms, converged=bool(converged), steps=opt.nsteps)


def ts_optimize(
    atoms: Atoms,
    attach: Optional[AttachFn] = None,
    order: int = 1,
    fmax: float = 0.05,
    steps: int = 200,
    logfile: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> TSOptimizationResult:
    """Transition state optimization using Sella.

    Parameters
    ----------
    atoms:
        Initial TS guess (modified in-place).
    attach:
        AttachFn to set the calculator.
    order:
        Number of negative eigenvalues to follow. 1 for a first-order TS.
    fmax, steps, logfile, trajectory:
        Same as geometry_optimize.
    """
    from sella import Sella

    _ensure_calc(atoms, attach)
    opt = Sella(atoms, order=order, logfile=logfile, trajectory=trajectory)
    converged = opt.run(fmax=fmax, steps=steps)
    if not converged:
        print("[ts_optimize] WARNING: did not converge")
    return TSOptimizationResult(atoms=atoms, converged=bool(converged), steps=opt.nsteps)


def run_neb(
    reactant: Atoms,
    product: Atoms,
    attach: Optional[AttachFn] = None,
    n_images: int = 15,
    k: float = 0.1,
    climb: bool = True,
    method: str = "aseneb",
    fmax: float = 0.05,
    steps: int = 1000,
    pre_optimize: bool = True,
    trajectory: Optional[str] = None,
) -> NEBResult:
    """NEB pathway search with IDPP interpolation.

    Parameters
    ----------
    reactant, product:
        Endpoint structures. Copied internally; originals are not modified.
    attach:
        AttachFn applied to all intermediate images (and endpoints if
        ``pre_optimize=True``).
    n_images:
        Total number of images including the two endpoints.
    k:
        Spring constant (eV/Å²).
    climb:
        Enable climbing-image NEB.
    method:
        NEB method string passed to ASE NEB (``'aseneb'``, ``'eb'``, …).
    pre_optimize:
        Pre-optimize endpoints before running NEB.
    """
    from ase.mep import NEB
    from ase.mep.neb import NEBOptimizer

    if attach is None and (reactant.calc is None or product.calc is None):
        raise ValueError("Provide an attach_fn or pre-attach calculators to endpoints.")

    if pre_optimize and attach is not None:
        _pre_optimize_endpoints(reactant, product, attach)

    images = (
        [reactant.copy()]
        + [reactant.copy() for _ in range(n_images - 2)]
        + [product.copy()]
    )
    neb = NEB(images, k=k, climb=climb, method=method)
    neb.interpolate(method="idpp")

    if attach is not None:
        attach(images)

    opt = NEBOptimizer(neb, trajectory=trajectory)
    converged = opt.run(fmax=fmax, steps=steps)
    if not converged:
        print("[run_neb] WARNING: did not converge")

    return NEBResult(
        images=images,
        converged=bool(converged),
        n_images=n_images,
        method=method,
    )


def run_dmf(
    reactant: Atoms,
    product: Atoms,
    attach: Optional[AttachFn] = None,
    nmove: int = 20,
    update_teval: bool = True,
    convergence: str = "tight",
    pre_optimize: bool = True,
    trajectory: Optional[str] = None,
) -> DMFResult:
    """DirectMaxFlux TS pathway search.

    Parameters
    ----------
    reactant, product:
        Endpoint structures. Copied internally.
    attach:
        AttachFn applied to all DMF images (and endpoints if
        ``pre_optimize=True``).
    nmove:
        Number of moves in the DMF algorithm.
    update_teval:
        Whether to update the transition-state evaluator during the run.
    convergence:
        Convergence tolerance string accepted by DMF (``'tight'``, …).
    pre_optimize:
        Pre-optimize endpoints before running DMF.
    """
    from dmf import DirectMaxFlux, interpolate_fbenm

    if pre_optimize and attach is not None:
        _pre_optimize_endpoints(reactant, product, attach)

    ref_images = [reactant.copy(), product.copy()]
    fbenm = interpolate_fbenm(ref_images, correlated=True)
    coefs = fbenm.coefs.copy()

    mxflx = DirectMaxFlux(ref_images, coefs=coefs, nmove=nmove, update_teval=update_teval)

    if attach is not None:
        attach(mxflx.images)

    mxflx.solve(tol=convergence)

    final_images = list(mxflx.images)
    if attach is not None:
        attach(final_images)
    for atoms in final_images:
        atoms.get_forces()

    if trajectory is not None:
        ase_write(trajectory, final_images)

    return DMFResult(images=final_images, nmove=nmove, convergence=convergence)


def run_irc(
    ts_atoms: Atoms,
    attach: Optional[AttachFn] = None,
    fmax: float = 0.1,
    steps: int = 300,
    dx: float = 0.05,
    eta: float = 1e-4,
    gamma: float = 0.4,
    strict: bool = True,
    forward_traj: Optional[str] = None,
    reverse_traj: Optional[str] = None,
) -> IRCResult:
    """Intrinsic reaction coordinate (IRC) calculation.

    Runs IRC in both directions from the TS and assembles a full path:
    ``reactant side → TS → product side``.

    Parameters
    ----------
    ts_atoms:
        Transition state structure.
    attach:
        AttachFn to set the calculator.
    fmax:
        Force convergence criterion (eV/Å).
    steps:
        Maximum steps per direction.
    dx, eta, gamma:
        Sella IRC parameters.
    strict:
        Raise RuntimeError if either direction fails to converge.
    forward_traj, reverse_traj:
        Optional .traj file paths for each direction.
    """
    from sella import IRC
    from ase.io import Trajectory

    _ensure_calc(ts_atoms, attach)

    def _run_direction(direction: str, traj_file: Optional[str]):
        atoms_copy = ts_atoms.copy()
        atoms_copy.calc = ts_atoms.calc
        opt = IRC(atoms_copy, trajectory=traj_file, dx=dx, eta=eta, gamma=gamma)
        converged = opt.run(fmax=fmax, steps=steps, direction=direction)
        if strict and not converged:
            raise RuntimeError(f"[run_irc] IRC {direction} direction did not converge")

        if traj_file is not None and Path(traj_file).exists():
            with Trajectory(traj_file) as traj:
                path = list(traj)
        else:
            path = [atoms_copy]

        # Drop the TS frame if it appears as the first image
        if path and np.allclose(path[0].positions, ts_atoms.positions, atol=1e-3):
            path = path[1:]

        return path, bool(converged)

    forward_path, forward_ok = _run_direction("forward", forward_traj)
    reverse_path, reverse_ok = _run_direction("reverse", reverse_traj)

    ts_atoms.get_forces()
    reverse_path.reverse()
    full_path = reverse_path + [ts_atoms] + forward_path

    return IRCResult(
        path=full_path,
        forward_converged=forward_ok,
        reverse_converged=reverse_ok,
    )


def freq_analysis(
    atoms: Atoms,
    attach: Optional[AttachFn] = None,
    name: str = "vib",
    delta: float = 0.01,
) -> FreqResult:
    """Vibrational frequency analysis using finite differences.

    Parameters
    ----------
    atoms:
        Structure at a stationary point (optimized geometry or TS).
    attach:
        AttachFn to set the calculator.
    name:
        Prefix for intermediate files written by ASE Vibrations.
    delta:
        Finite-difference displacement (Å).
    """
    from ase.vibrations import Vibrations

    _ensure_calc(atoms, attach)
    vib = Vibrations(atoms, delta=delta, name=name)
    vib.run()

    freqs = vib.get_frequencies()           # complex array; imaginary → negative real
    freqs_real = freqs.real                 # cm⁻¹
    hessian = vib.get_vibrations().get_hessian_2d()
    n_imag = int(np.sum(freqs_real < -1.0)) # threshold: -1 cm⁻¹

    return FreqResult(
        frequencies=freqs_real.tolist(),
        n_imaginary=n_imag,
        hessian=hessian,
    )


def run_mcd(
    atoms: Atoms,
    constraints: dict,
    attach: Optional[AttachFn] = None,
    n_relax: int = 5,
    fmax: float = 0.05,
    optimizer: str = "BFGS",
) -> MCDResult:
    """Deform a reactant toward a product geometry via Multi-Coordinate Driving.

    MCD drives selected internal coordinates (bonds, angles, dihedrals) toward
    target values in small incremental steps, relaxing all other degrees of
    freedom at each step.  This is the correct way to generate a product
    structure from a reactant — it avoids the random-conformation noise that
    arises from separately embedding each structure from SMILES.

    The input ``atoms`` object should originate from ``atoms_from_sdf`` so
    that ``atoms.info["connectivity"]`` is populated.  Bond info is propagated
    to every frame in the returned pathway so the trajectory can be exported
    as a bond-preserving SDF with ``images_to_sdf``.

    Parameters
    ----------
    atoms:
        Starting geometry (reactant).  Must have ``atoms.info["connectivity"]``
        set (i.e. come from ``atoms_from_sdf``) for the output SDF to carry
        correct bond orders.  Modified internally on a copy.
    constraints:
        Dict specifying which internal coordinates to drive and how far.
        Key tuple length determines coordinate type:

        - 2-tuple ``(i, j)``       → bond distance, target in Å
        - 3-tuple ``(i, j, k)``    → valence angle, target in degrees
        - 4-tuple ``(i, j, k, l)`` → dihedral angle, target in degrees

        Value is ``(target_value, n_steps)``.

        Example — SN2-like concerted bond change::

            constraints = {
                (0, 5): (4.0, 20),   # break bond 0-5, stretch to 4.0 Å in 20 steps
                (2, 5): (1.5, 20),   # form bond 2-5, contract to 1.5 Å in 20 steps
            }

    attach:
        AttachFn to set the calculator.  Used only if ``atoms.calc`` is None.
    n_relax:
        Number of geometry relaxation steps (perpendicular DOF) per MCD step.
    fmax:
        Force convergence criterion for the constrained relaxation (eV/Å).
    optimizer:
        ASE optimizer for the constrained relaxation: ``"BFGS"``, ``"FIRE"``,
        or ``"LBFGS"``.

    Returns
    -------
    MCDResult
        Contains the full pathway (``List[Atoms]``), energy profile, TS guess
        (highest-energy frame), and forward/reverse barriers.
    """
    from asemcd import MCD

    working = atoms.copy()
    _ensure_calc(working, attach)

    mcd_obj = MCD(working, logfile=None)
    raw = mcd_obj.scan(
        constraints,
        n_relax=n_relax,
        fmax=fmax,
        optimizer=optimizer,
        save_trajectory=False,
    )

    # Propagate bond connectivity to every pathway frame so images_to_sdf works
    connectivity = atoms.info.get("connectivity")
    if connectivity is not None:
        for frame in raw["pathway"]:
            if "connectivity" not in frame.info:
                frame.info["connectivity"] = connectivity

    return MCDResult(
        pathway=raw["pathway"],
        energies=np.asarray(raw["energies"]),
        ts_index=int(raw["ts_index"]),
        ts_atoms=raw["ts_atoms"],
        forward_barrier=float(raw["forward_barrier"]),
        reverse_barrier=float(raw["reverse_barrier"]),
    )


def run_popcornn(
    reactant: Atoms,
    product: Atoms,
    attach: Optional[AttachFn] = None,
    potential: str = "mace",
    model_name: str = "mace-mp-0",
    n_embed: int = 1,
    depth: int = 2,
    num_record_points: int = 101,
    optimization_stages: Optional[List[dict]] = None,
    pre_optimize: bool = False,
) -> PopcornnResult:
    """Neural-network continuous path optimization using Popcornn.

    Represents the reaction path as a continuous neural network function and
    optimises it against MLIP potentials.  Supports multi-stage optimisation:
    e.g. geodesic interpolation for initial de-clashing, then MLIP-based
    variational reaction energy minimisation.

    This is an alternative string method to NEB / DMF — useful for smooth
    reaction paths and when NEB convergence is slow.

    Parameters
    ----------
    reactant, product:
        Endpoint structures.  Copied internally; originals not modified.
    attach:
        Optional AttachFn used only for endpoint pre-optimisation
        (``pre_optimize=True``).  Popcornn manages its own MLIP
        potentials internally.
    potential:
        MLIP potential name for the refinement stage.  Options include
        ``'mace'``, ``'uma'``, ``'chgnet'``, ``'orb'``, etc.
    model_name:
        Model checkpoint name (e.g. ``'mace-mp-0'``, ``'uma-s-1'``).
    n_embed:
        NN path embedding dimensions.
    depth:
        MLP depth of the NN path representation.
    num_record_points:
        Number of discrete points sampled along the optimised continuous
        path for the returned ``images`` list.
    optimization_stages:
        Explicit list of stage dicts for :meth:`Popcornn.optimize_path`.
        Each dict should contain ``potential_params``, ``integrator_params``,
        ``optimizer_params``, and ``num_optimizer_iterations``.
        If ``None``, a sensible two-stage default is used (geodesic
        interpolation → MLIP refinement).
    pre_optimize:
        If ``True`` and ``attach`` is given, pre-optimise endpoints with
        ASE before running Popcornn.
    """
    from popcornn import Popcornn

    r = reactant.copy()
    p = product.copy()

    if pre_optimize and attach is not None:
        _pre_optimize_endpoints(r, p, attach)

    # Transfer charge/spin info to endpoint images
    charge = int(reactant.info.get("charge", 0))
    spin = int(reactant.info.get("spin", 1))
    for img in (r, p):
        img.info["charge"] = charge
        img.info["spin"] = spin

    path = Popcornn(
        images=[r, p],
        path_params={"name": "mlp", "n_embed": n_embed, "depth": depth},
        num_record_points=num_record_points,
    )

    if optimization_stages is None:
        optimization_stages = [
            {
                "potential_params": {"potential": "repel"},
                "integrator_params": {"path_ode_names": "geodesic"},
                "optimizer_params": {"optimizer": {"name": "adam", "lr": 0.1}},
                "num_optimizer_iterations": 1000,
            },
            {
                "potential_params": {
                    "potential": potential,
                    "model_name": model_name,
                },
                "integrator_params": {
                    "path_ode_names": "projected_variational_reaction_energy",
                },
                "optimizer_params": {"optimizer": {"name": "adam", "lr": 0.001}},
                "num_optimizer_iterations": 1000,
            },
        ]

    final_images, ts_image = path.optimize_path(*optimization_stages)
    if not isinstance(final_images, list):
        final_images = list(final_images)

    # Extract energies — popcornn may store them in atoms.info or via calc
    energies = []
    for img in final_images:
        if hasattr(img, "info") and "energy" in img.info:
            energies.append(float(img.info["energy"]))
        elif img.calc is not None:
            try:
                energies.append(float(img.get_potential_energy()))
            except Exception:
                energies.append(0.0)
        else:
            energies.append(0.0)
    energies = np.asarray(energies)

    if len(energies) > 0:
        ts_e = float(energies.max())
        forward_barrier = ts_e - float(energies[0])
        reverse_barrier = ts_e - float(energies[-1])
    else:
        forward_barrier = 0.0
        reverse_barrier = 0.0

    # Propagate bond connectivity from reactant to all frames
    connectivity = reactant.info.get("connectivity")
    if connectivity is not None:
        for img in final_images:
            img.info.setdefault("connectivity", connectivity)
        if hasattr(ts_image, "info"):
            ts_image.info.setdefault("connectivity", connectivity)

    return PopcornnResult(
        images=final_images,
        ts_atoms=ts_image,
        energies=energies,
        forward_barrier=forward_barrier,
        reverse_barrier=reverse_barrier,
    )
