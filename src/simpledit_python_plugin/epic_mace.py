import os
import tempfile
import traceback
from typing import List, Optional
from pydantic import BaseModel

# Import Opsin for name-to-smiles conversion
from simpledit_python_plugin.opsin import name_to_structure, OpsinRequest

# Try to import mace
try:
    import mace
    from mace import Complex
    MACE_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: Failed to import mace: {e}")
    MACE_AVAILABLE = False
except Exception as e:
    print(f"DEBUG: Unexpected error importing mace: {e}")
    MACE_AVAILABLE = False

class EpicMaceRequest(BaseModel):
    metal: str
    oxidation_state: int = 0
    geometry: str = "octahedral"
    ligands: List[str]  # Can be names or SMILES
    
    # Advanced options
    name: Optional[str] = "complex"
    res_structs: int = 1
    regime: str = "all"
    get_enantiomers: bool = False
    trans_cycle: bool = False
    mer_rule: bool = True
    num_confs: int = 10
    rms_thresh: float = 1.0
    num_repr_confs: int = 3
    e_rel_max: float = 25.0
    drop_close_energy: bool = True
    only_global_minimum: bool = False

class EpicMaceResponse(BaseModel):
    sdf: Optional[str] = None
    error: Optional[str] = None
    logs: Optional[str] = None

def generate_complex(request: EpicMaceRequest) -> EpicMaceResponse:
    if not MACE_AVAILABLE:
        return EpicMaceResponse(error="epic-mace library not found. Please install it via pip.")

    # 1. Convert ligands to SMILES
    ligands_smiles = []
    for ligand in request.ligands:
        if any(c in ligand for c in "*=[]#"):
            ligands_smiles.append(ligand)
        else:
            opsin_req = OpsinRequest(name=ligand)
            opsin_res = name_to_structure(opsin_req)
            if opsin_res.result and isinstance(opsin_res.result, str):
                ligands_smiles.append(opsin_res.result)
            else:
                ligands_smiles.append(ligand)

    try:
        from mace import ComplexFromLigands
        from rdkit import Chem
        import io

        # 2. Create Complex
        # Note: res_structs maps to maxResonanceStructures
        complex_obj = ComplexFromLigands(
            ligands=ligands_smiles, 
            CA=request.metal, 
            geom=request.geometry,
            maxResonanceStructures=request.res_structs
        )

        # 3. Get Stereomers
        stereomers = complex_obj.GetStereomers(
            regime=request.regime,
            dropEnantiomers=not request.get_enantiomers, 
            minTransCycle=None if not request.trans_cycle else 5, 
            merRule=request.mer_rule
        )

        if not stereomers:
            return EpicMaceResponse(error="No stereomers found for the given input.")

        # List to store (energy, stereomer_obj, conf_id, stereomer_index)
        all_conformers = []

        # 4. Generate Conformers for each stereomer
        for i, stereomer in enumerate(stereomers):
            # Add Conformers
            stereomer.AddConformers(
                numConfs=request.num_confs,
                rmsThresh=request.rms_thresh
            )
            
            # Filter Representative Conformers
            stereomer.GetRepresentativeConfs(
                numConfs=request.num_repr_confs,
                dE=request.e_rel_max,
                dropCloseEnergy=request.drop_close_energy
            )
            
            # Collect conformers and their energies
            if hasattr(stereomer, 'mol3D') and stereomer.mol3D:
                for conf in stereomer.mol3D.GetConformers():
                    conf_id = conf.GetId()
                    # Try to get energy. 
                    # Based on grep, GetConfEnergy(confId) exists.
                    try:
                        energy = stereomer.GetConfEnergy(conf_id)
                    except:
                        energy = 0.0 # Fallback if energy not available
                    
                    all_conformers.append({
                        "energy": energy,
                        "stereomer": stereomer,
                        "conf_id": conf_id,
                        "stereomer_idx": i
                    })

        if not all_conformers:
             return EpicMaceResponse(error="Stereomers found but conformer generation failed (no 3D structures).")

        # 5. Filter for Global Minimum if requested
        if request.only_global_minimum:
            # Sort by energy (ascending) and take the first one
            all_conformers.sort(key=lambda x: x["energy"])
            all_conformers = [all_conformers[0]]

        # 6. Export to SDF
        sio = io.StringIO()
        writer = Chem.SDWriter(sio)
        
        for item in all_conformers:
            stereomer = item["stereomer"]
            conf_id = item["conf_id"]
            stereomer_idx = item["stereomer_idx"]
            energy = item["energy"]
            
            # Add properties
            stereomer.mol3D.SetProp("StereomerID", str(stereomer_idx))
            stereomer.mol3D.SetProp("Energy", str(energy))
            
            writer.write(stereomer.mol3D, confId=conf_id)
            
        writer.close()
        final_sdf = sio.getvalue()
        
        return EpicMaceResponse(sdf=final_sdf)

    except Exception as e:
        traceback.print_exc()
        return EpicMaceResponse(error=f"Internal Error: {str(e)}")
