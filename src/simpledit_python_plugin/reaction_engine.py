"""
Reaction SMARTS 적용 엔진

반응물 SMILES + Reaction SMARTS → 생성물 SMILES
RDKit AllChem.ReactionFromSmarts 사용.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem


class ApplyReactionRequest(BaseModel):
    reactant_smiles: List[str]          # 반응물 SMILES 리스트
    smarts: str                         # Reaction SMARTS (e.g. "[C:1][Cl].[OH:2]>>[C:1][O:2].[Cl-]")
    return_all_products: bool = False   # True: 모든 product set 반환, False: 첫 번째만


class ApplyReactionResponse(BaseModel):
    product_smiles: List[str]           # 생성물 SMILES 리스트 (단일 반응 결과)
    all_products: Optional[List[List[str]]] = None  # return_all_products=True일 때
    error: Optional[str] = None


def apply_reaction(req: ApplyReactionRequest) -> ApplyReactionResponse:
    try:
        rxn = AllChem.ReactionFromSmarts(req.smarts)
        if rxn is None:
            return ApplyReactionResponse(
                product_smiles=[], error=f"Invalid reaction SMARTS: {req.smarts}"
            )

        reactants = []
        for smi in req.reactant_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return ApplyReactionResponse(
                    product_smiles=[], error=f"Invalid SMILES: {smi}"
                )
            reactants.append(mol)

        products_sets = rxn.RunReactants(tuple(reactants))

        if not products_sets:
            return ApplyReactionResponse(
                product_smiles=[],
                error="Reaction SMARTS did not match reactants. Check atom mapping."
            )

        # 첫 번째 product set sanitize 및 변환
        sanitized = []
        for p in products_sets[0]:
            try:
                Chem.SanitizeMol(p)
                sanitized.append(Chem.MolToSmiles(p))
            except Exception:
                sanitized.append(Chem.MolToSmiles(p))

        result = ApplyReactionResponse(product_smiles=sanitized)

        if req.return_all_products:
            all_sets = []
            for pset in products_sets:
                ps = []
                for p in pset:
                    try:
                        Chem.SanitizeMol(p)
                        ps.append(Chem.MolToSmiles(p))
                    except Exception:
                        ps.append("")
                all_sets.append(ps)
            result.all_products = all_sets

        return result

    except Exception as e:
        return ApplyReactionResponse(product_smiles=[], error=str(e))
