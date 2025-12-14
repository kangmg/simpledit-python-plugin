# TOON Format API Quick Start

## Quick Usage

### Option 1: Direct Function Call (No Server Needed)

```python
from simpledit_python_plugin.toon_format import convert_to_toon_format, ToonFormatRequest
from rdkit import Chem

# Create molecule
smiles = "CCO.O"  # Ethanol + Water
mol = Chem.MolFromSmiles(smiles)
sdf = Chem.MolToMolBlock(mol)

# Convert to TOON format
request = ToonFormatRequest(
    sdf=sdf,
    selected_indices=[0, 1, 2],
    method="efgs"  # or "accfg"
)

result = convert_to_toon_format(request)
print(result.toon_output)
```

### Option 2: API Call (Server Required)

**Start server:**
```bash
simpledit-py
```

**Send request:**
```python
import requests
from rdkit import Chem

smiles = "CCO.O"
mol = Chem.MolFromSmiles(smiles)
sdf = Chem.MolToMolBlock(mol)

response = requests.post(
    "http://localhost:8000/api/python/toon-format",
    json={
        "sdf": sdf,
        "selected_indices": [0, 1, 2],
        "method": "efgs"
    }
)

print(response.json()["toon_output"])
```

**Or with curl:**
```bash
curl -X POST http://localhost:8000/api/python/toon-format \
  -H "Content-Type: application/json" \
  -d '{
    "sdf": "...",
    "selected_indices": [0, 1, 2],
    "method": "efgs"
  }'
```

## Output Format

```
fragments[2|]{fragment_idx|name|mapped_smiles|fragment_atoms_idx}:
  0|fragment_0|"[C:1]([C:2]([O:3][H:10])..."|"[0, 1, 2, 4, 5, 6, 7, 8, 9]"
  1|fragment_1|"[O:4]([H:11])[H:12]"|"[3, 10, 11]"
atoms[12|]{atom_idx|symbol|connected_atom_idx|coordinate|fragment_idx|local_context|selected}:
  0|C|"[1, 4, 5, 6]"|"[0.0, 0.0, 0.0]"|0|"C (SP3), connected to [C(1), H(4), H(5), H(6)]"|true
  1|C|"[0, 2, 7, 8]"|"[1.3, 0.75, 0.0]"|0|"C (SP3), connected to [C(0), H(7), H(8), O(2)]"|true
  ...
```

## Request Parameters

- `sdf` (string, required): SDF format molecule
- `selected_indices` (list[int], optional): Selected atom indices (0-indexed)
- `method` (string, optional): "efgs" (default) or "accfg"

## Response

- `toon_output` (string): TOON format text with pipe delimiter
- `error` (string | null): Error message if any

## Example Script

Run `example_toon_api.py` for full examples:
```bash
uv run python example_toon_api.py
```
