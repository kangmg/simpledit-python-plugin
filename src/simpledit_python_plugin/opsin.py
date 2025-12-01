from typing import List, Union, Optional
from pydantic import BaseModel

try:
    from py2opsin import py2opsin
except ImportError:
    py2opsin = None

class OpsinRequest(BaseModel):
    name: Union[str, List[str]]
    output_format: str = "SMILES"
    allow_acid: bool = True
    allow_radicals: bool = True
    allow_bad_stereo: bool = False
    wildcard_radicals: bool = True

class OpsinResponse(BaseModel):
    result: Union[str, List[str], bool]
    error: Optional[str] = None

def name_to_structure(request: OpsinRequest) -> OpsinResponse:
    if py2opsin is None:
        return OpsinResponse(result=False, error="py2opsin not installed")
    
    # Check for Java availability
    import shutil
    import subprocess
    if shutil.which("java") is None:
        return OpsinResponse(result=False, error="Java Runtime Environment (JRE) not found. Please install Java 8+.")
    
    try:
        subprocess.run(["java", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return OpsinResponse(result=False, error="Java found but failed to run. Please install Java 8+.")
    except Exception:
        return OpsinResponse(result=False, error="Error checking Java version.")

    try:
        # py2opsin returns False on error, or the string/list on success
        result = py2opsin(
            chemical_name=request.name,
            output_format=request.output_format,
            allow_acid=request.allow_acid,
            allow_radicals=request.allow_radicals,
            allow_bad_stereo=request.allow_bad_stereo,
            wildcard_radicals=request.wildcard_radicals
        )
        
        if result is False:
             return OpsinResponse(result=False, error="OPSIN failed to parse name")
             
        return OpsinResponse(result=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return OpsinResponse(result=False, error=f"Internal Error: {str(e)}")
