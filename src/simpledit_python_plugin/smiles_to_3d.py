"""
SMILES → 3D SDF 변환

RDKit ETKDG로 3D 좌표 생성 후 MMFF 힘장 최적화.
에이전트가 반응물/생성물 SMILES를 알고 있을 때 즉시 3D SDF로 변환하는 용도.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem


class SmilesTo3DRequest(BaseModel):
    smiles: str
    add_hs: bool = True           # 수소 추가 여부
    optimize_ff: bool = True      # MMFF 힘장 최적화 여부
    random_seed: int = 42


class SmilesTo3DResponse(BaseModel):
    sdf: str
    n_atoms: int
    error: Optional[str] = None


def smiles_to_3d(req: SmilesTo3DRequest) -> SmilesTo3DResponse:
    try:
        mol = Chem.MolFromSmiles(req.smiles)
        if mol is None:
            return SmilesTo3DResponse(sdf="", n_atoms=0, error=f"Invalid SMILES: {req.smiles}")

        if req.add_hs:
            mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = req.random_seed
        result = AllChem.EmbedMolecule(mol, params)

        if result == -1:
            # ETKDGv3 실패 시 random coords fallback
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if req.optimize_ff:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            except Exception:
                pass  # MMFF 실패해도 좌표 자체는 반환

        sdf = Chem.MolToMolBlock(mol)
        return SmilesTo3DResponse(sdf=sdf, n_atoms=mol.GetNumAtoms())

    except Exception as e:
        return SmilesTo3DResponse(sdf="", n_atoms=0, error=str(e))
