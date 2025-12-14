"""
Calculation module for ASE-based computational chemistry

Provides interfaces for:
- Geometry optimization
- NEB (Nudged Elastic Band)
- Frequency calculations
- IRC (Intrinsic Reaction Coordinate)
- Single point energy
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.vibrations import Vibrations
from ase.neb import NEB
from rdkit import Chem
import io


class CalculationRequest(BaseModel):
    """Base request model for calculations"""
    sdf: str = Field(..., description="SDF format molecule")
    calculator: str = Field(..., description="Calculator name (mace, xt b, etc.)")
    calculator_params: Dict[str, Any] = Field(default={}, description="Calculator parameters")


class OptimizationRequest(CalculationRequest):
    """Request model for geometry optimization"""
    optimizer: str = Field(default="BFGS", description="Optimizer (BFGS, LBFGS, FIRE)")
    fmax: float = Field(default=0.05, description="Force convergence criterion (eV/Å)")
    steps: int = Field(default=200, description="Maximum optimization steps")


class OptimizationResponse(BaseModel):
    """Response model for geometry optimization"""
    optimized_sdf: str = Field(..., description="Optimized structure in SDF format")
    energy: float = Field(..., description="Final energy (eV)")
    converged: bool = Field(..., description="Whether optimization converged")
    steps_taken: int = Field(..., description="Number of steps taken")
    error: Optional[str] = Field(None, description="Error message if any")


class FrequencyRequest(CalculationRequest):
    """Request model for frequency calculation"""
    delta: float = Field(default=0.01, description="Displacement for finite differences (Å)")


class FrequencyResponse(BaseModel):
    """Response model for frequency calculation"""
    frequencies: List[float] = Field(..., description="Vibrational frequencies (cm⁻¹)")
    ir_intensities: Optional[List[float]] = Field(None, description="IR intensities")
    zero_point_energy: float = Field(..., description="Zero-point energy (eV)")
    error: Optional[str] = Field(None, description="Error message if any")


class NEBRequest(BaseModel):
    """Request model for NEB calculation"""
    initial_sdf: str = Field(..., description="Initial structure SDF")
    final_sdf: str = Field(..., description="Final structure SDF")
    n_images: int = Field(default=7, description="Number of intermediate images")
    calculator: str = Field(..., description="Calculator name")
    calculator_params: Dict[str, Any] = Field(default={}, description="Calculator parameters")
    fmax: float = Field(default=0.05, description="Force convergence (eV/Å)")
    steps: int = Field(default=200, description="Maximum steps")


class NEBResponse(BaseModel):
    """Response model for NEB calculation"""
    pathway_sdfs: List[str] = Field(..., description="SDFs of structures along pathway")
    energies: List[float] = Field(..., description="Energies along pathway (eV)")
    barrier_forward: float = Field(..., description="Forward barrier (eV)")
    barrier_reverse: float = Field(..., description="Reverse barrier (eV)")
    converged: bool = Field(..., description="Whether NEB converged")
    error: Optional[str] = Field(None, description="Error message if any")


class SinglePointRequest(CalculationRequest):
    """Request model for single point energy calculation"""
    pass


class SinglePointResponse(BaseModel):
    """Response model for single point calculation"""
    energy: float = Field(..., description="Total energy (eV)")
    forces: Optional[List[List[float]]] = Field(None, description="Atomic forces (eV/Å)")
    error: Optional[str] = Field(None, description="Error message if any")


def sdf_to_atoms(sdf: str) -> Atoms:
    """Convert SDF string to ASE Atoms object"""
    mol = Chem.MolFromMolBlock(sdf)
    if mol is None:
        raise ValueError("Failed to parse SDF")
    
    # Add hydrogens if not present
    mol = Chem.AddHs(mol)
    
    # Get 3D coordinates
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=42)
    
    conf = mol.GetConformer()
    
    # Create ASE Atoms
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    positions = [[p.x, p.y, p.z] for p in positions]
    
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def atoms_to_sdf(atoms: Atoms) -> str:
    """Convert ASE Atoms to SDF string"""
    # Write to temporary xyz, then convert to SDF via RDKit
    from io import StringIO
    from ase.io import write as ase_write
    
    # Create XYZ string
    xyz_io = StringIO()
    ase_write(xyz_io, atoms, format='xyz')
    xyz_str = xyz_io.getvalue()
    
    # Convert XYZ to RDKit mol
    mol = Chem.MolFromXYZBlock(xyz_str)
    if mol is None:
        # Fallback: create mol from scratch
        mol = Chem.RWMol()
        for symbol in atoms.get_chemical_symbols():
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)
        
        conf = Chem.Conformer(len(atoms))
        for i, pos in enumerate(atoms.positions):
            conf.SetAtomPosition(i, tuple(pos))
        mol.AddConformer(conf)
    
    # Convert to SDF
    sdf = Chem.MolToMolBlock(mol)
    return sdf


def get_calculator_instance(calculator_name: str, params: Dict[str, Any]):
    """Get calculator instance by name"""
    calculator_name = calculator_name.lower()
    
    if calculator_name == 'mace':
        from .calculators.mace import MACECalculator
        calc_wrapper = MACECalculator()
        return calc_wrapper.get_calculator(**params)
    elif calculator_name in ['xtb', 'gfn2-xtb']:
        from .calculators.xtb import XTBCalculator
        calc_wrapper = XTBCalculator()
        return calc_wrapper.get_calculator(**params)
    else:
        raise ValueError(f"Unknown calculator: {calculator_name}")


# Calculation functions (to be implemented)
def run_optimization(request: OptimizationRequest) -> OptimizationResponse:
    """Run geometry optimization"""
    # TODO: Implement
    return OptimizationResponse(
        optimized_sdf="",
        energy=0.0,
        converged=False,
        steps_taken=0,
        error="Not implemented yet"
    )


def run_frequency(request: FrequencyRequest) -> FrequencyResponse:
    """Run frequency calculation"""
    # TODO: Implement
    return FrequencyResponse(
        frequencies=[],
        zero_point_energy=0.0,
        error="Not implemented yet"
    )


def run_neb(request: NEBRequest) -> NEBResponse:
    """Run NEB calculation"""
    # TODO: Implement
    return NEBResponse(
        pathway_sdfs=[],
        energies=[],
        barrier_forward=0.0,
        barrier_reverse=0.0,
        converged=False,
        error="Not implemented yet"
    )


def run_single_point(request: SinglePointRequest) -> SinglePointResponse:
    """Run single point energy calculation"""
    # TODO: Implement
    return SinglePointResponse(
        energy=0.0,
        error="Not implemented yet"
    )
