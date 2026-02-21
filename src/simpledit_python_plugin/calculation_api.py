"""
FastAPI request/response models and endpoint handlers for calculation.py

Each handler:
  1. Deserializes the SDF input via atoms_from_sdf (bond info embedded)
  2. Constructs an AttachFn from the requested calculator
  3. Calls the relevant core calculation function
  4. Serializes output back to SDF via atoms_to_sdf (bonds preserved)
  5. Returns a typed Pydantic response

Errors are returned as 200 with a non-None ``error`` field so that the
api_client's retry-on-network-error logic is not triggered for calculation
failures.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class CalculatorParams(BaseModel):
    """Common calculator parameters shared by all endpoints."""
    charge: int = 0
    spin: int = 1          # spin multiplicity: 1=singlet, 2=doublet, 3=triplet


_OPTIMIZER_MAP = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}


def _make_attach(calculator: str, params: CalculatorParams):
    """Build an AttachFn from calculator name + params."""
    from .calculation import make_attach_fn
    from .calculators.registry import get_calculator

    charge, spin = params.charge, params.spin
    # Capture by value with a default-arg trick
    return make_attach_fn(
        lambda c=calculator, ch=charge, s=spin: get_calculator(c, charge=ch, spin=s)
    )


# ---------------------------------------------------------------------------
# 1. Geometry optimization
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    sdf: str
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    optimizer: str = "BFGS"   # BFGS | LBFGS | FIRE
    fmax: float = 0.05
    steps: int = 200


class OptimizeResponse(BaseModel):
    optimized_sdf: str
    energy: float             # eV
    converged: bool
    steps_taken: int
    error: Optional[str] = None


def handle_optimize(req: OptimizeRequest) -> OptimizeResponse:
    try:
        from .calculation import atoms_from_sdf, atoms_to_sdf, geometry_optimize

        atoms = atoms_from_sdf(req.sdf)
        attach = _make_attach(req.calculator, req.calculator_params)
        opt_cls = _OPTIMIZER_MAP.get(req.optimizer.upper(), BFGS)
        result = geometry_optimize(
            atoms, attach=attach, optimizer=opt_cls,
            fmax=req.fmax, steps=req.steps,
        )
        return OptimizeResponse(
            optimized_sdf=atoms_to_sdf(result.atoms),
            energy=float(result.atoms.get_potential_energy()),
            converged=result.converged,
            steps_taken=result.steps,
        )
    except Exception as exc:
        return OptimizeResponse(
            optimized_sdf=req.sdf, energy=0.0,
            converged=False, steps_taken=0, error=str(exc),
        )


# ---------------------------------------------------------------------------
# 2. TS optimization
# ---------------------------------------------------------------------------

class TSOptimizeRequest(BaseModel):
    sdf: str
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    order: int = 1            # number of negative eigenvalues to follow
    fmax: float = 0.05
    steps: int = 200


class TSOptimizeResponse(BaseModel):
    optimized_sdf: str
    energy: float
    converged: bool
    steps_taken: int
    error: Optional[str] = None


def handle_ts_optimize(req: TSOptimizeRequest) -> TSOptimizeResponse:
    try:
        from .calculation import atoms_from_sdf, atoms_to_sdf, ts_optimize

        atoms = atoms_from_sdf(req.sdf)
        attach = _make_attach(req.calculator, req.calculator_params)
        result = ts_optimize(
            atoms, attach=attach, order=req.order,
            fmax=req.fmax, steps=req.steps,
        )
        return TSOptimizeResponse(
            optimized_sdf=atoms_to_sdf(result.atoms),
            energy=float(result.atoms.get_potential_energy()),
            converged=result.converged,
            steps_taken=result.steps,
        )
    except Exception as exc:
        return TSOptimizeResponse(
            optimized_sdf=req.sdf, energy=0.0,
            converged=False, steps_taken=0, error=str(exc),
        )


# ---------------------------------------------------------------------------
# 3. NEB
# ---------------------------------------------------------------------------

class NEBRequest(BaseModel):
    initial_sdf: str
    final_sdf: str
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    n_images: int = 15
    k: float = 0.1
    climb: bool = True
    fmax: float = 0.05
    steps: int = 1000
    pre_optimize: bool = True


class NEBResponse(BaseModel):
    pathway_sdfs: List[str]   # one SDF block per image
    energies: List[float]     # eV per image
    barrier_forward: float    # eV  (relative to image[0])
    barrier_reverse: float    # eV  (relative to image[-1])
    converged: bool
    error: Optional[str] = None


def handle_neb(req: NEBRequest) -> NEBResponse:
    try:
        from .calculation import atoms_from_sdf, atoms_to_sdf, run_neb

        reactant = atoms_from_sdf(req.initial_sdf)
        product  = atoms_from_sdf(req.final_sdf)
        attach   = _make_attach(req.calculator, req.calculator_params)

        result = run_neb(
            reactant, product, attach=attach,
            n_images=req.n_images, k=req.k, climb=req.climb,
            fmax=req.fmax, steps=req.steps,
            pre_optimize=req.pre_optimize,
        )

        # Propagate reactant bond connectivity to intermediate images
        connectivity = reactant.info.get("connectivity")
        if connectivity:
            for img in result.images:
                img.info.setdefault("connectivity", connectivity)

        energies = [float(img.get_potential_energy()) for img in result.images]
        ts_e = max(energies)
        return NEBResponse(
            pathway_sdfs=[atoms_to_sdf(img) for img in result.images],
            energies=energies,
            barrier_forward=ts_e - energies[0],
            barrier_reverse=ts_e - energies[-1],
            converged=result.converged,
        )
    except Exception as exc:
        return NEBResponse(
            pathway_sdfs=[], energies=[],
            barrier_forward=0.0, barrier_reverse=0.0,
            converged=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# 4. MCD (Multi-Coordinate Driving)
# ---------------------------------------------------------------------------

class MCDConstraintItem(BaseModel):
    """Single internal coordinate constraint for MCD."""
    indices: List[int]    # 2 → distance (Å), 3 → angle (°), 4 → dihedral (°)
    target: float
    n_steps: int = 20


class MCDRequest(BaseModel):
    sdf: str
    constraints: List[MCDConstraintItem]
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    n_relax: int = 5
    fmax: float = 0.05
    optimizer: str = "BFGS"


class MCDResponse(BaseModel):
    product_sdf: str          # last frame of pathway
    pathway_sdfs: List[str]   # all frames
    ts_sdf: str               # highest-energy frame (TS guess)
    energies: List[float]     # eV per frame
    forward_barrier: float
    reverse_barrier: float
    error: Optional[str] = None


def handle_mcd(req: MCDRequest) -> MCDResponse:
    try:
        from .calculation import atoms_from_sdf, atoms_to_sdf, run_mcd

        atoms  = atoms_from_sdf(req.sdf)
        attach = _make_attach(req.calculator, req.calculator_params)

        # Convert list → dict expected by asemcd
        constraints = {
            tuple(c.indices): (c.target, c.n_steps)
            for c in req.constraints
        }

        result = run_mcd(
            atoms, constraints=constraints, attach=attach,
            n_relax=req.n_relax, fmax=req.fmax, optimizer=req.optimizer,
        )

        return MCDResponse(
            product_sdf=atoms_to_sdf(result.pathway[-1]),
            pathway_sdfs=[atoms_to_sdf(img) for img in result.pathway],
            ts_sdf=atoms_to_sdf(result.ts_atoms),
            energies=result.energies.tolist(),
            forward_barrier=result.forward_barrier,
            reverse_barrier=result.reverse_barrier,
        )
    except Exception as exc:
        return MCDResponse(
            product_sdf=req.sdf, pathway_sdfs=[], ts_sdf=req.sdf,
            energies=[], forward_barrier=0.0, reverse_barrier=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# 5. MCD constraint generation (bond graph comparison)
# ---------------------------------------------------------------------------

class MCDConstraintsRequest(BaseModel):
    reactant_sdf: str
    product_sdf: str
    n_steps: int = 20                 # default steps per constraint
    breaking_distance: float = 3.5    # Å target for breaking bonds


class MCDConstraintsResponse(BaseModel):
    constraints: List[MCDConstraintItem]
    description: str      # human-readable summary of detected bond changes


def handle_mcd_constraints(req: MCDConstraintsRequest) -> MCDConstraintsResponse:
    try:
        from .mcd_constraints import generate_mcd_constraints
        return generate_mcd_constraints(req)
    except Exception as exc:
        return MCDConstraintsResponse(constraints=[], description=f"Error: {exc}")


# ---------------------------------------------------------------------------
# 6. IRC
# ---------------------------------------------------------------------------

class IRCRequest(BaseModel):
    sdf: str              # TS structure
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    fmax: float = 0.1
    steps: int = 300
    dx: float = 0.05


class IRCResponse(BaseModel):
    path_sdfs: List[str]     # reactant side → TS → product side
    forward_converged: bool
    reverse_converged: bool
    error: Optional[str] = None


def handle_irc(req: IRCRequest) -> IRCResponse:
    try:
        from .calculation import atoms_from_sdf, atoms_to_sdf, run_irc

        ts_atoms = atoms_from_sdf(req.sdf)
        attach   = _make_attach(req.calculator, req.calculator_params)
        result   = run_irc(
            ts_atoms, attach=attach,
            fmax=req.fmax, steps=req.steps, dx=req.dx,
            strict=False,
        )

        connectivity = ts_atoms.info.get("connectivity")
        if connectivity:
            for img in result.path:
                img.info.setdefault("connectivity", connectivity)

        return IRCResponse(
            path_sdfs=[atoms_to_sdf(img) for img in result.path],
            forward_converged=result.forward_converged,
            reverse_converged=result.reverse_converged,
        )
    except Exception as exc:
        return IRCResponse(
            path_sdfs=[], forward_converged=False,
            reverse_converged=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# 7. Vibrational frequency analysis
# ---------------------------------------------------------------------------

class FreqRequest(BaseModel):
    sdf: str
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)
    delta: float = 0.01


class FreqResponse(BaseModel):
    frequencies: List[float]    # cm⁻¹; imaginary shown as negative
    n_imaginary: int
    zero_point_energy: float    # eV
    error: Optional[str] = None


def handle_freq(req: FreqRequest) -> FreqResponse:
    try:
        from .calculation import atoms_from_sdf, freq_analysis

        atoms  = atoms_from_sdf(req.sdf)
        attach = _make_attach(req.calculator, req.calculator_params)
        result = freq_analysis(atoms, attach=attach, delta=req.delta)

        freqs = np.array(result.frequencies)
        zpe = float(np.sum(freqs[freqs > 0]) * 0.5 * 1.239842e-4)  # cm⁻¹ → eV

        return FreqResponse(
            frequencies=result.frequencies,
            n_imaginary=result.n_imaginary,
            zero_point_energy=zpe,
        )
    except Exception as exc:
        return FreqResponse(
            frequencies=[], n_imaginary=0,
            zero_point_energy=0.0, error=str(exc),
        )


# ---------------------------------------------------------------------------
# 8. Single-point energy
# ---------------------------------------------------------------------------

class SinglePointRequest(BaseModel):
    sdf: str
    calculator: str = "xtb"
    calculator_params: CalculatorParams = Field(default_factory=CalculatorParams)


class SinglePointResponse(BaseModel):
    energy: float
    forces: Optional[List[List[float]]] = None
    error: Optional[str] = None


def handle_single_point(req: SinglePointRequest) -> SinglePointResponse:
    try:
        from .calculation import atoms_from_sdf

        atoms  = atoms_from_sdf(req.sdf)
        attach = _make_attach(req.calculator, req.calculator_params)
        attach(atoms)

        return SinglePointResponse(
            energy=float(atoms.get_potential_energy()),
            forces=atoms.get_forces().tolist(),
        )
    except Exception as exc:
        return SinglePointResponse(energy=0.0, error=str(exc))
