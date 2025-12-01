import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Simpledit Extra")

# Enable CORS for development (allowing frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "simpledit-extra"}

from simpledit_python_plugin.opsin import name_to_structure, OpsinRequest, OpsinResponse

@app.get("/api/capabilities")
async def get_capabilities():
    """
    Returns the list of available Python modules.
    This helps the frontend decide which features to enable.
    """
    modules = []
    try:
        import py2opsin
        modules.append("opsin")
    except ImportError:
        pass

    return {
        "python": True,
        "modules": modules
    }

@app.post("/api/python/opsin", response_model=OpsinResponse)
async def run_opsin(request: OpsinRequest):
    return name_to_structure(request)

from simpledit_python_plugin.epic_mace import generate_complex, EpicMaceRequest, EpicMaceResponse

@app.post("/api/python/epic-mace/generate", response_model=EpicMaceResponse)
async def run_epic_mace(request: EpicMaceRequest):
    return generate_complex(request)

def start():
    """Entry point for the CLI command 'simpledit-py'"""
    import argparse
    parser = argparse.ArgumentParser(description="Simpledit Python Backend")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    print(f"Starting Simpledit Python Backend on port {args.port}...")
    uvicorn.run("simpledit_extra.main:app", host="127.0.0.1", port=args.port, reload=True)

if __name__ == "__main__":
    start()
