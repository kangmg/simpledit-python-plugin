import os
import uuid
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Simpledit Python Plugin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health + capabilities
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "simpledit-python-plugin"}


@app.get("/api/capabilities")
async def get_capabilities():
    """List available modules so the frontend can enable features."""
    modules = []
    for mod, name in [
        ("py2opsin", "opsin"),
        ("accfg",    "toon-format"),
        ("xtb",      "xtb"),
        ("mace",     "mace"),
        ("sella",    "ts-optimize"),
        ("asemcd",   "mcd"),
        ("popcornn", "popcornn"),
        ("dmf",      "dmf"),
    ]:
        try:
            __import__(mod)
            modules.append(name)
        except ImportError:
            pass
    return {"python": True, "modules": modules}


# ---------------------------------------------------------------------------
# Existing endpoints (opsin, epic-mace, toon-format)
# ---------------------------------------------------------------------------

from simpledit_python_plugin.opsin import name_to_structure, OpsinRequest, OpsinResponse

@app.post("/api/python/opsin", response_model=OpsinResponse)
async def run_opsin(request: OpsinRequest):
    return name_to_structure(request)


from simpledit_python_plugin.epic_mace import generate_complex, EpicMaceRequest, EpicMaceResponse

@app.post("/api/python/epic-mace/generate", response_model=EpicMaceResponse)
async def run_epic_mace(request: EpicMaceRequest):
    return generate_complex(request)


from simpledit_python_plugin.toon_format import convert_to_toon_format, ToonFormatRequest, ToonFormatResponse

@app.post("/api/python/toon-format", response_model=ToonFormatResponse)
async def run_toon_format(request: ToonFormatRequest):
    return convert_to_toon_format(request)


# ---------------------------------------------------------------------------
# Calculation endpoints
# ---------------------------------------------------------------------------

from simpledit_python_plugin.calculation_api import (
    OptimizeRequest,    OptimizeResponse,    handle_optimize,
    TSOptimizeRequest,  TSOptimizeResponse,  handle_ts_optimize,
    NEBRequest,         NEBResponse,         handle_neb,
    MCDRequest,         MCDResponse,         handle_mcd,
    MCDConstraintsRequest, MCDConstraintsResponse, handle_mcd_constraints,
    IRCRequest,         IRCResponse,         handle_irc,
    FreqRequest,        FreqResponse,        handle_freq,
    SinglePointRequest, SinglePointResponse, handle_single_point,
    PopcornnRequest,    PopcornnResponse,    handle_popcornn,
    DMFRequest,         DMFResponse,         handle_dmf,
)


@app.post("/api/python/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """Geometry optimization (BFGS / LBFGS / FIRE)."""
    return handle_optimize(request)


@app.post("/api/python/ts-optimize", response_model=TSOptimizeResponse)
async def ts_optimize(request: TSOptimizeRequest):
    """Transition-state optimization with Sella."""
    return handle_ts_optimize(request)


@app.post("/api/python/neb", response_model=NEBResponse)
async def neb(request: NEBRequest):
    """NEB pathway search with IDPP interpolation."""
    return handle_neb(request)


@app.post("/api/python/mcd", response_model=MCDResponse)
async def mcd(request: MCDRequest):
    """Deform reactant toward product via Multi-Coordinate Driving."""
    return handle_mcd(request)


@app.post("/api/python/mcd/constraints", response_model=MCDConstraintsResponse)
async def mcd_constraints(request: MCDConstraintsRequest):
    """Auto-generate MCD constraints by comparing reactant and product bond graphs."""
    return handle_mcd_constraints(request)


@app.post("/api/python/irc", response_model=IRCResponse)
async def irc(request: IRCRequest):
    """IRC from a TS structure (forward + reverse, assembled as full path)."""
    return handle_irc(request)


@app.post("/api/python/frequency", response_model=FreqResponse)
async def frequency(request: FreqRequest):
    """Vibrational frequency analysis via finite differences."""
    return handle_freq(request)


@app.post("/api/python/single-point", response_model=SinglePointResponse)
async def single_point(request: SinglePointRequest):
    """Single-point energy + forces."""
    return handle_single_point(request)


@app.post("/api/python/popcornn", response_model=PopcornnResponse)
async def popcornn(request: PopcornnRequest):
    """NN continuous path optimization (alternative string method to NEB/DMF)."""
    return handle_popcornn(request)


@app.post("/api/python/dmf", response_model=DMFResponse)
async def dmf(request: DMFRequest):
    """DirectMaxFlux TS pathway search (alternative to NEB)."""
    return handle_dmf(request)


# ---------------------------------------------------------------------------
# SMILES and reaction utilities
# ---------------------------------------------------------------------------

from simpledit_python_plugin.smiles_to_3d import (
    SmilesTo3DRequest, SmilesTo3DResponse, smiles_to_3d
)

@app.post("/api/python/smiles-to-3d", response_model=SmilesTo3DResponse)
async def convert_smiles_to_3d(request: SmilesTo3DRequest):
    """Convert SMILES string to 3D SDF using RDKit ETKDG + MMFF."""
    return smiles_to_3d(request)


from simpledit_python_plugin.reaction_engine import (
    ApplyReactionRequest, ApplyReactionResponse, apply_reaction
)

@app.post("/api/python/apply-reaction", response_model=ApplyReactionResponse)
async def run_apply_reaction(request: ApplyReactionRequest):
    """Apply reaction SMARTS to reactant SMILES, return product SMILES."""
    return apply_reaction(request)


# ---------------------------------------------------------------------------
# SDF structure store
#
# Lightweight in-memory key-value store so the agent can deposit an SDF
# (e.g. after a simpledit export) and later retrieve it by ID.
# ---------------------------------------------------------------------------

_structure_store: Dict[str, str] = {}


@app.post("/api/structures/store")
async def store_structure(body: dict):
    """Store an SDF string, return a short ID.

    Body: {"sdf": "<sdf_string>"}
    """
    sdf = body.get("sdf")
    if not sdf:
        raise HTTPException(status_code=422, detail="'sdf' field is required")
    sid = uuid.uuid4().hex[:8]
    _structure_store[sid] = sdf
    return {"id": sid}


@app.get("/api/structures/{structure_id}")
async def get_structure(structure_id: str):
    """Retrieve a stored SDF by ID."""
    sdf = _structure_store.get(structure_id)
    if sdf is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")
    return {"id": structure_id, "sdf": sdf}


@app.delete("/api/structures/{structure_id}")
async def delete_structure(structure_id: str):
    """Remove a stored SDF."""
    if structure_id not in _structure_store:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")
    _structure_store.pop(structure_id)
    return {"deleted": structure_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start():
    """CLI entry point: simpledit-py [--port PORT]"""
    import argparse
    parser = argparse.ArgumentParser(description="Simpledit Python Plugin")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Starting Simpledit Python Plugin on port {args.port}...")
    uvicorn.run(
        "simpledit_python_plugin.main:app",
        host="127.0.0.1", port=args.port, reload=True,
    )


if __name__ == "__main__":
    start()
