"""
Calculator Registry

Dynamically imports available calculators and provides unified interface.
Handles parameter name mapping (e.g., spin vs multiplicity).
"""

from typing import Dict, Any, Optional, Callable
from ase.calculators.calculator import Calculator


# Available calculators registry
CALCULATORS: Dict[str, Callable] = {}

# Parameter name mappings for different calculators
# Maps from unified names to calculator-specific names
PARAM_MAPPINGS = {
    'xtb': {
        'spin': 'uhf',  # XTB uses 'uhf' for spin
    },
    'mace': {
        # MACE doesn't use spin/charge directly
    },
}


def register_calculator(name: str, get_calc_func: Callable):
    """Register a calculator"""
    CALCULATORS[name] = get_calc_func


def map_parameters(calculator_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map unified parameter names to calculator-specific names
    
    Args:
        calculator_name: Name of calculator
        params: Parameters with unified names (spin, charge, etc.)
        
    Returns:
        Parameters with calculator-specific names
    """
    if calculator_name not in PARAM_MAPPINGS:
        return params
    
    mapping = PARAM_MAPPINGS[calculator_name]
    mapped_params = params.copy()
    
    for unified_name, calc_specific_name in mapping.items():
        if unified_name in mapped_params:
            value = mapped_params.pop(unified_name)
            mapped_params[calc_specific_name] = value
    
    return mapped_params


def get_calculator(name: str, charge: int = 0, spin: int = 1, **kwargs) -> Calculator:
    """
    Get calculator instance with unified interface
    
    Args:
        name: Calculator name
        charge: Molecular charge (default: 0)
        spin: Spin multiplicity (default: 1, singlet)
        **kwargs: Additional calculator-specific parameters
        
    Returns:
        ASE Calculator instance
    """
    name = name.lower()
    
    if name not in CALCULATORS:
        available = ', '.join(CALCULATORS.keys())
        raise ValueError(
            f"Calculator '{name}' not available. "
            f"Available calculators: {available}"
        )
    
    # Prepare parameters with charge and spin
    params = {'charge': charge, 'spin': spin, **kwargs}
    
    # Map to calculator-specific parameter names
    params = map_parameters(name, params)
    
    # Get calculator
    get_calc = CALCULATORS[name]
    return get_calc(**params)


def list_available_calculators() -> Dict[str, Dict[str, Any]]:
    """
    List all available calculators with their default parameters
    
    Returns:
        Dictionary of calculator info
    """
    info = {}
    for name, get_calc in CALCULATORS.items():
        # Try to get default params
        try:
            default_params = get_calc.__defaults__ or {}
            info[name] = {
                'available': True,
                'default_params': default_params
            }
        except:
            info[name] = {'available': True}
    
    return info


# Try to import calculators
try:
    from .mace import get_mace_calculator, DEFAULT_PARAMS as MACE_DEFAULTS
    register_calculator('mace', get_mace_calculator)
except ImportError:
    pass

try:
    from .xtb import get_xtb_calculator, DEFAULT_PARAMS as XTB_DEFAULTS
    register_calculator('xtb', get_xtb_calculator)
except ImportError:
    pass

# Add more calculators here as they are implemented
