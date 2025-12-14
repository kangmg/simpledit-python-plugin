"""
Calculator Template

Template for creating new calculator implementations.
Copy this file and modify for your calculator.

IMPORTANT:
- Must define DEFAULT_PARAMS dictionary
- Must accept charge and spin parameters
- Function name should be get_<calculator_name>_calculator
"""

from typing import Dict, Any
from ase.calculators.calculator import Calculator


# Default parameters for this calculator
DEFAULT_PARAMS = {
    'method': 'default_method',  # Replace with actual default
    # Add other calculator-specific defaults here
}


def get_template_calculator(
    charge: int = 0,
    spin: int = 1,
    method: str = DEFAULT_PARAMS['method'],
    **kwargs
) -> Calculator:
    """
    Get Template calculator instance
    
    REQUIRED PARAMETERS (always present):
        charge: Molecular charge (default: 0)
        spin: Spin multiplicity (default: 1 for singlet)
            - 1 = singlet (0 unpaired electrons)
            - 2 = doublet (1 unpaired electron)
            - 3 = triplet (2 unpaired electrons)
    
    CALCULATOR-SPECIFIC PARAMETERS:
        method: Calculation method (default: 'default_method')
        **kwargs: Additional parameters
    
    Returns:
        ASE Calculator instance
    
    Example:
        >>> calc = get_template_calculator(charge=0, spin=1)
        >>> calc = get_template_calculator(charge=1, spin=2, method='special')
    """
    try:
        # Import calculator (replace with actual import)
        # from some_package import SomeCalculator
        
        # Create calculator instance
        # Note: Some calculators may need parameter name mapping
        # For example:
        #   - charge might be 'charge' or 'molecular_charge'
        #   - spin might be 'multiplicity', 'uhf', 'spin_multiplicity', etc.
        
        calculator = None  # Replace with actual calculator
        
        # Example for a real calculator:
        # calculator = SomeCalculator(
        #     charge=charge,
        #     multiplicity=spin,  # or uhf, or spin_multiplicity
        #     method=method,
        #     **kwargs
        # )
        
        return calculator
        
    except ImportError as e:
        raise ImportError(
            f"Template calculator not installed. "
            f"Install with: pip install template-package"
        ) from e


# After implementing, add to registry.py:
#
# try:
#     from .template import get_template_calculator, DEFAULT_PARAMS as TEMPLATE_DEFAULTS
#     register_calculator('template', get_template_calculator)
# except ImportError:
#     pass
