"""
XTB Calculator

Extended Tight Binding semi-empirical quantum chemistry
"""

from typing import Dict, Any
from ase.calculators.calculator import Calculator


# Default parameters
DEFAULT_PARAMS = {
    'method': 'GFN2-xTB',  # GFN0-xTB, GFN1-xTB, GFN2-xTB
}


def get_xtb_calculator(
    charge: int = 0,
    spin: int = 1,
    method: str = DEFAULT_PARAMS['method'],
    **kwargs
) -> Calculator:
    """
    Get XTB calculator instance
    
    Args:
        charge: Molecular charge
        spin: Spin multiplicity (converted to uhf for XTB)
        method: XTB method (GFN0-xTB, GFN1-xTB, GFN2-xTB)
        **kwargs: Additional XTB parameters
        
    Returns:
        XTB Calculator instance
        
    Note:
        XTB uses 'uhf' parameter instead of 'spin'
        uhf = number of unpaired electrons = spin - 1
    """
    try:
        from xtb.ase.calculator import XTB
        
        # Convert spin multiplicity to uhf (unpaired electrons)
        uhf = spin - 1
        
        calculator = XTB(
            method=method,
            charge=charge,
            uhf=uhf,
            **kwargs
        )
        
        return calculator
        
    except ImportError as e:
        raise ImportError(
            "XTB is not installed. "
            "Install with: pip install xtb-python"
        ) from e
