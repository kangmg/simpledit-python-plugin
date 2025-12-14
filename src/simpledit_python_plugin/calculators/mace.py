"""
MACE Calculator

Machine-learning Accelerated Computational Engine
"""

from typing import Dict, Any
from ase.calculators.calculator import Calculator


# Default parameters
DEFAULT_PARAMS = {
    'model': 'medium',  # small, medium, large
    'device': 'cpu',     # cpu or cuda
}


def get_mace_calculator(
    charge: int = 0,
    spin: int = 1,
    model: str = DEFAULT_PARAMS['model'],
    device: str = DEFAULT_PARAMS['device'],
    **kwargs
) -> Calculator:
    """
    Get MACE calculator instance
    
    Args:
        charge: Molecular charge (not used by MACE, kept for interface)
        spin: Spin multiplicity (not used by MACE, kept for interface)
        model: MACE model size (small, medium, large)
        device: Device (cpu or cuda)
        **kwargs: Additional parameters
        
    Returns:
        MACE Calculator instance
    """
    try:
        from mace.calculators import mace_mp
        
        # Note: MACE-MP-0 doesn't use charge/spin directly
        # It's a general-purpose force field
        calculator = mace_mp(
            model=model,
            device=device,
            default_dtype='float64',
            **kwargs
        )
        
        return calculator
        
    except ImportError as e:
        raise ImportError(
            "MACE is not installed. "
            "Install with: pip install mace-torch"
        ) from e
