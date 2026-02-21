"""
Calculation module for ASE + MLIP based computational chemistry

Architecture
------------
Core functions take ASE Atoms directly and return typed dataclasses.
Serialization (Atoms <-> SDF/XYZ/SMILES) lives in a separate section and
is only used at the API boundary (FastAPI endpoints in main.py).

    Local use:  atoms → core_fn(atoms, attach_fn) → Result
    API use:    payload → deserialize → core_fn → Result → serialize → response

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
# ---------------------------------------------------------------------------

def atoms_from_xyz(xyz: str) -> Atoms:
    """Parse XYZ string into ASE Atoms."""
    return ase_read(StringIO(xyz), format="xyz")


def atoms_to_xyz(atoms: Atoms) -> str:
    """Serialize ASE Atoms to XYZ string."""
    buf = StringIO()
    ase_write(buf, atoms, format="xyz")
    return buf.getvalue()


def atoms_from_sdf(sdf: str) -> Atoms:
    """Parse SDF/MOL block into ASE Atoms (via RDKit for bond info)."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(sdf, removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("Failed to parse SDF block")
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


def atoms_to_sdf(atoms: Atoms) -> str:
    """Serialize ASE Atoms to SDF block (via RDKit)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    xyz = atoms_to_xyz(atoms)
    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        raise ValueError("Failed to convert Atoms to RDKit Mol")
    AllChem.DetermineBonds(mol)
    return Chem.MolToMolBlock(mol)


def smiles_to_atoms(smiles: str, add_hydrogens: bool = True, seed: int = 42) -> Atoms:
    """Convert a SMILES string to a 3D ASE Atoms object using RDKit ETKDG.

    Parameters
    ----------
    smiles:
        SMILES string.
    add_hydrogens:
        Whether to add explicit hydrogens before embedding.
    seed:
        Random seed for the conformer generator.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        raise RuntimeError(f"RDKit could not generate a 3D conformer for: {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


def traj_to_xyz_list(images: List[Atoms]) -> List[str]:
    """Serialize a list of Atoms to a list of XYZ strings."""
    return [atoms_to_xyz(a) for a in images]


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
