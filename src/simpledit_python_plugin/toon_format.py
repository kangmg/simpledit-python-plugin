"""
Toon Format Converter - Clean implementation
Converts SDF format molecules to TOON format with functional group analysis.
Uses toon_format package for output formatting with pipe delimiter.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem
from toon_format import encode, EncodeOptions



class ToonFormatRequest(BaseModel):
    """Request model for toon format conversion"""
    sdf: str = Field(..., description="SDF format string of the molecule")
    selected_indices: List[int] = Field(default=[], description="List of selected atom indices")


class ToonFormatResponse(BaseModel):
    """Response model for toon format conversion"""
    toon_output: str = Field(..., description="TOON format output with pipe delimiter")
    error: Optional[str] = Field(None, description="Error message if any")


def extract_functional_groups_accfg(mol: Chem.Mol) -> List[Dict[str, Any]]:
    """Extract functional groups using AccFG library"""
    try:
        from accfg import AccFG
        
        accfg = AccFG()
        smiles = Chem.MolToSmiles(mol)
        results = accfg.run(smiles, show_atoms=True, show_graph=False, canonical=True)
        
        functional_groups = []
        if results:
            for fg_name, instances in results.items():
                for atom_indices in instances:
                    functional_groups.append({
                        'name': fg_name,
                        'atom_indices': atom_indices
                    })
        
        return functional_groups
    except:
        return []


def generate_local_context(mol: Chem.Mol, atom_idx: int, accfg_groups: List[Dict], fragment_name: Optional[str] = None) -> str:
    """
    Generate local context description for an atom
    
    Format examples:
    - "C (SP3) in benzene, connected to [C(1), H(4), H(5), H(6)]"
    - "H on C(1) in primary hydroxyl"
    - "H on C(0)"
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    
    # Find AccFG group this atom belongs to
    group_name = None
    for group in accfg_groups:
        if atom_idx in group['atom_indices']:
            group_name = group['name']
            break
    
    # Get neighbors with indices
    neighbor_atoms = atom.GetNeighbors()
    neighbor_strs = [f"{n.GetSymbol()}({n.GetIdx()})" for n in neighbor_atoms]
    
    # Build context
    if symbol == 'H':
        # For hydrogen - show parent with index
        if neighbor_atoms:
            parent = neighbor_atoms[0]
            parent_symbol = parent.GetSymbol()
            parent_idx = parent.GetIdx()
            context = f"H on {parent_symbol}({parent_idx})"
            
            # If H is not in a group, check if its parent is
            if not group_name:
                for group in accfg_groups:
                    if parent_idx in group['atom_indices']:
                        group_name = group['name']
                        break
            
            # Only add group name if it's a real AccFG group
            if group_name:
                context += f" in {group_name}"
        else:
            context = "H"
    else:
        # For other atoms - show neighbors with indices
        hybridization = str(atom.GetHybridization()).replace("HybridizationType.", "")
        
        # Sort neighbors by index for consistent output
        neighbor_strs_sorted = sorted(neighbor_strs)
        neighbors_str = f"[{', '.join(neighbor_strs_sorted)}]"
        
        if group_name:
            context = f"{symbol} ({hybridization}) in {group_name}, connected to {neighbors_str}"
        else:
            context = f"{symbol} ({hybridization}), connected to {neighbors_str}"
    
    return context


def convert_to_toon_format(request: ToonFormatRequest) -> ToonFormatResponse:
    """
    Convert SDF format to TOON format with functional group analysis
    
    Uses AccFG for functional group detection and context enrichment.
    
    Returns TOON format string with pipe delimiter, including:
    - fragments table: fragment information
    - atoms table: atom information with fragment_idx
    """
    try:
        # Parse SDF
        mol = Chem.MolFromMolBlock(request.sdf)
        if mol is None:
            return ToonFormatResponse(
                toon_output="",
                error="Failed to parse SDF structure"
            )
        
        # Add explicit hydrogens
        mol_with_h = Chem.AddHs(mol)
        
        # Get 3D coordinates (or generate if needed)
        if mol_with_h.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
        
        conf = mol_with_h.GetConformer()
        
        # Separate into fragments
        frag_mols = Chem.GetMolFrags(mol_with_h, asMols=True, sanitizeFrags=False)
        frag_atom_indices = Chem.GetMolFrags(mol_with_h)  # Tuple of tuples with atom indices
        
        # Extract AccFG groups for context enrichment (using original molecule)
        accfg_groups = extract_functional_groups_accfg(mol_with_h)
        
        # Build fragments data
        fragments_data = []
        for frag_idx, (frag_mol, frag_atoms) in enumerate(zip(frag_mols, frag_atom_indices)):
            # Generate mapped SMILES for this fragment
            # Add atom mapping to fragment
            frag_mol_copy = Chem.Mol(frag_mol)
            for atom in frag_mol_copy.GetAtoms():
                # Map to original molecule atom index
                orig_idx = frag_atoms[atom.GetIdx()]
                atom.SetAtomMapNum(orig_idx + 1)  # 1-indexed
            
            mapped_smiles = Chem.MolToSmiles(frag_mol_copy, allHsExplicit=True, canonical=False)
            
            # Get molecular formula
            from rdkit.Chem import Descriptors
            formula = Descriptors.rdMolDescriptors.CalcMolFormula(frag_mol)
            
            fragments_data.append({
                'fragment_idx': frag_idx,
                'formula': formula,
                'mapped_smiles': mapped_smiles,
                'fragment_atoms_idx': str(list(frag_atoms))
            })
        
        # Build atom-to-fragment mapping
        atom_to_fragment = {}
        for frag_idx, frag_atoms in enumerate(frag_atom_indices):
            for atom_idx in frag_atoms:
                atom_to_fragment[atom_idx] = frag_idx
        
        # Build atoms data
        atoms_data = []
        num_atoms = mol_with_h.GetNumAtoms()
        
        for atom_idx in range(num_atoms):
            atom = mol_with_h.GetAtomWithIdx(atom_idx)
            pos = conf.GetAtomPosition(atom_idx)
            
            # Get connected atoms
            connected = [n.GetIdx() for n in atom.GetNeighbors()]
            
            # Get fragment formula for this atom
            fragment_idx = atom_to_fragment.get(atom_idx, 0)
            frag_formula = fragments_data[fragment_idx]['formula'] if fragment_idx < len(fragments_data) else None
            
            # Generate local context (note: we don't pass fragment formula to context)
            local_context = generate_local_context(mol_with_h, atom_idx, accfg_groups, None)
            
            # Is selected?
            selected = atom_idx in request.selected_indices
            
            # Get fragment index
            fragment_idx = atom_to_fragment.get(atom_idx, 0)
            
            atoms_data.append({
                'atom_idx': atom_idx,
                'symbol': atom.GetSymbol(),
                'connected_atom_idx': str(connected),
                'coordinate': str([round(pos.x, 2), round(pos.y, 2), round(pos.z, 2)]),
                'fragment_idx': fragment_idx,
                'local_context': local_context,
                'selected': selected
            })
        
        # Combine fragments and atoms for TOON output
        toon_data = {
            'fragments': fragments_data,
            'atoms': atoms_data
        }
        
        # Convert to TOON format with pipe delimiter
        toon_output = encode(toon_data, {'delimiter': 'pipe'})
        
        return ToonFormatResponse(
            toon_output=toon_output,
            error=None
        )
        
    except Exception as e:
        import traceback
        return ToonFormatResponse(
            toon_output="",
            error=f"Error converting to toon format: {str(e)}\n{traceback.format_exc()}"
        )
