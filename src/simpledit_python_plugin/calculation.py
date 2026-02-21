"""
Calculation module for ASE + MLIP based computational chemistry

Architecture
------------
Core functions take ASE Atoms directly and return typed dataclasses.
Serialization (Atoms <-> SDF) lives in a separate section and is only
used at the API boundary (FastAPI endpoints in main.py).

    Local use:  atoms → core_fn(atoms, attach_fn) → Result
    API use:    sdf → atoms_from_sdf → core_fn → Result → atoms_to_sdf(template) → sdf

SDF is the canonical I/O format because it preserves:
  - Explicit bond orders and bond types
  - 3D coordinates
  - Fragment/disconnected-component information

IMPORTANT — bond preservation after ASE computation
----------------------------------------------------
ASE Atoms objects carry only element symbols + positions; bond info is
not stored internally.  After optimization / NEB / IRC the positions
change, but the bond topology is unchanged.

The correct round-trip is:
  1.  sdf_to_mol_and_atoms(sdf)  → (rdkit_mol, atoms)   # bonds live in mol
  2.  core_fn(atoms, ...)        → Result (new positions in Result.atoms)
  3.  atoms_to_sdf(result.atoms, template_mol=rdkit_mol) # coords updated, bonds kept

Never reconstruct bonds from geometry via DetermineBonds on a post-
computation structure; bond inference can silently produce wrong topology.

Supported calculations
----------------------
- Geometry optimization  : geometry_optimize  (BFGS / LBFGS / FIRE)
- TS optimization        : ts_optimize        (Sella)
- Double-ended search    : run_neb            (NEB + IDPP)
                           run_dmf            (DirectMaxFlux)
- IRC                    : run_irc            (Sella IRC, forward + reverse)
- Vibrational analysis   : freq_analysis      (ASE Vibrations)
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
# SDF is the canonical format: it preserves bond orders, bond types, and
# fragment topology across the computation round-trip.
#
# Recommended pattern
# -------------------
#   mol, atoms = sdf_to_mol_and_atoms(sdf)      # bonds stay in `mol`
#   result     = geometry_optimize(atoms, ...)   # positions updated in-place
#   out_sdf    = atoms_to_sdf(result.atoms, mol) # coords replaced, bonds kept
#
# For multi-fragment SDF (reactant + product in one file, salts, etc.)
#   blocks = split_sdf(sdf)                      # split on "$$$$"
#   mols_atoms = [sdf_to_mol_and_atoms(b) for b in blocks]
# ---------------------------------------------------------------------------

def _mol_from_sdf(sdf: str):
    """Return a sanitized RDKit Mol from an SDF block (with H atoms)."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(sdf, removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("Failed to parse SDF block")
    return mol


def _mol_to_atoms(mol) -> Atoms:
    """Extract element symbols + 3D positions from an RDKit Mol into ASE Atoms."""
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


# -- Primary entry points ----------------------------------------------------

def sdf_to_mol_and_atoms(sdf: str):
    """Parse a single SDF block, returning (rdkit_mol, ase_atoms).

    The RDKit Mol retains the original bond topology and should be kept as a
    template for ``atoms_to_sdf`` after computation so bonds are not lost.

    Parameters
    ----------
    sdf:
        A single MOL/SDF block (one molecule).

    Returns
    -------
    mol : rdkit.Chem.Mol
        Molecule with explicit bond information.  Use as ``template_mol`` in
        ``atoms_to_sdf`` to preserve bonds after ASE computation.
    atoms : ase.Atoms
        Positions + elements only; suitable for ASE calculators.
    """
    mol = _mol_from_sdf(sdf)
    return mol, _mol_to_atoms(mol)


def atoms_from_sdf(sdf: str) -> Atoms:
    """Parse a single SDF block into ASE Atoms (positions only).

    Use ``sdf_to_mol_and_atoms`` instead when you need to write SDF back after
    computation, to avoid bond-inference errors.
    """
    return _mol_to_atoms(_mol_from_sdf(sdf))


def atoms_to_sdf(atoms: Atoms, template_mol=None) -> str:
    """Serialize ASE Atoms back to an SDF block.

    Parameters
    ----------
    atoms:
        Structure with (possibly updated) positions.
    template_mol:
        The RDKit Mol returned by ``sdf_to_mol_and_atoms``.  When provided,
        only the atomic coordinates are updated — bond orders, bond types, and
        all other molecular properties are preserved from the original.

        When *not* provided, bonds are inferred from geometry via
        ``DetermineBonds``.  This is a lossy fallback; avoid it for
        post-computation output where the original topology is available.
    """
    from rdkit import Chem

    if template_mol is not None:
        rw = Chem.RWMol(template_mol)
        conf = rw.GetConformer()
        for i, pos in enumerate(atoms.positions):
            conf.SetAtomPosition(i, pos.tolist())
        return Chem.MolToMolBlock(rw.GetMol())

    # Fallback: infer bonds from geometry (use only when no template is available)
    from rdkit.Chem import AllChem
    from io import StringIO as _StringIO
    buf = _StringIO()
    ase_write(buf, atoms, format="xyz")
    mol = Chem.MolFromXYZBlock(buf.getvalue())
    if mol is None:
        raise ValueError("Failed to convert Atoms to RDKit Mol")
    AllChem.DetermineBonds(mol)
    return Chem.MolToMolBlock(mol)


# -- Multi-fragment helpers --------------------------------------------------

def split_sdf(sdf: str) -> List[str]:
    """Split a multi-molecule SDF into a list of individual SDF blocks.

    SDF files use ``$$$$`` as a record separator.  This utility splits the
    file while keeping the terminator line attached to each block, so each
    returned string is a valid standalone SDF block.

    Useful for reactant/product pairs, salt forms, or any input where the
    caller packs multiple structures into one file.
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
    # Handle missing trailing separator
    remainder = "".join(current).strip()
    if remainder:
        blocks.append(remainder)
    return blocks


def atoms_list_from_sdf(sdf: str) -> List[Atoms]:
    """Parse a multi-molecule SDF into a list of ASE Atoms objects.

    Equivalent to ``[atoms_from_sdf(b) for b in split_sdf(sdf)]``.
    Use ``[sdf_to_mol_and_atoms(b) for b in split_sdf(sdf)]`` when bond
    preservation is needed after computation.
    """
    return [atoms_from_sdf(block) for block in split_sdf(sdf)]


def images_to_sdf(images: List[Atoms], template_mol=None) -> str:
    """Serialize a trajectory / NEB path to a multi-molecule SDF.

    Each frame becomes one SDF record, joined with ``$$$$`` separators.
    If ``template_mol`` is given, bond topology is preserved for every frame.

    Useful for exporting NEB images, IRC paths, or geometry optimization
    trajectories as a single SDF file.
    """
    blocks = [atoms_to_sdf(img, template_mol) for img in images]
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
