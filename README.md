# Simpledit Python Plugin

Python backend extensions for Simpledit.

## Requirements

- **Java 8+**: Required for `py2opsin` (IUPAC to SMILES conversion).

## Features

- **Opsin Integration**: Convert IUPAC names to SMILES using `py2opsin`.
- **Toon Format Conversion**: Convert SDF molecules to indexed SMILES with functional group analysis (AccFG & EFGs support).
- **Epic-MACE Integration**: Generate 3D metal complexes.

## Installation

```bash
pip install -e .
```

## Usage

This package is designed to be used as a backend service for the Simpledit CLI.
It can be started manually via:

```bash
simpledit-py
```

Or automatically via the main Simpledit CLI:

```bash
simpledit --python
```

**Default Port:** `8000` (Can be changed via `--port` argument)

### Interactive Documentation
Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)


## API Reference

### POST /api/python/opsin

Convert an IUPAC name to a chemical structure (SMILES).

**Request Body:**
```json
{
  "name": "ethylpentane",
  "output_format": "SMILES"  // Optional, default: "SMILES"
}
```

**Response:**
```json
{
  "result": "C(C)CCCCC",  // The SMILES string
  "error": null
}
```

**Notes:**
- Returns an empty string `""` if the name is valid but yields no result (e.g. some trivial names).

---

### POST /api/python/toon-format

Convert SDF format molecules to "toon format" with functional group analysis using either AccFG or EFGs method.

**Request Body:**
```json
{
  "sdf": "... SDF format string ...",
  "selected_indices": [0, 1, 2],  // Optional, default: []
  "method": "accfg"  // Optional, "accfg" or "efgs", default: "accfg"
}
```

**Response:**
```json
{
  "indexed_smiles": "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
  "functional_groups": [
    {
      "name": "primary hydroxyl",
      "smarts": null,
      "atom_indices": [2],
      "local_context": "FG atoms: O | Neighbors: C, H"
    }
  ],
  "selected": [true, true, false, false, false, false, false, false, false],
  "error": null
}
```

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sdf` | `string` | *required* | SDF format molecule string |
| `selected_indices` | `List[int]` | `[]` | Indices of selected atoms (0-indexed) |
| `method` | `string` | `"accfg"` | Functional group detection method: `"accfg"` or `"efgs"` |

**Methods Comparison:**

| Method | Description | Output Style | SMARTS Patterns |
| :--- | :--- | :--- | :--- |
| **accfg** | AccFG library - semantic, high-level functional groups | Natural language names (e.g., "primary hydroxyl", "carboxylic ester") | ❌ Not provided |
| **efgs** | EFGs (Ertl) - algorithmic SMARTS-based detection | Generic chemical names (e.g., "Hydroxyl", "Carbonyl") | ✅ Provided |

**Use Cases:**
- **AccFG**: Best for high-level functional group classification, drug-like molecule analysis
- **EFGs**: Best for detailed SMARTS-based analysis, comprehensive functional group detection

**Notes:**
- The `indexed_smiles` field contains SMILES with explicit hydrogens and atom mapping (1-indexed)
- Atom indices in `atom_indices` are 0-indexed and include explicit hydrogens
- The `selected` array is a boolean array matching the number of atoms (with explicit H)
- `local_context` describes the functional group atoms and their immediate neighbors

---

### POST /api/python/epic-mace/generate

Generate a 3D metal complex using `epic-mace`.

**Request Body:**
> **Note:** The `ligands` list supports both **SMILES strings** and **Chemical Names**. Names are automatically converted to SMILES using `py2opsin`.

```json
{
  "metal": "Fe",
  "oxidation_state": 2,
  "geometry": "octahedral",
  "ligands": ["pyridine", "Cl"],
  
  // Advanced Options (Defaults shown)
  "name": "complex",          // Name for the complex
  "res_structs": 1,           // Max resonance structures
  "regime": "all",            // Stereomer search regime: "CA", "ligands", or "all"
  "get_enantiomers": false,   // If true, keep both enantiomers (default: drop one)
  "trans_cycle": false,       // If true, allow trans-chelation (min size 5)
  "mer_rule": true,           // Apply empiric rule restricting rigid fragments
  "num_confs": 10,            // Number of conformers to generate per stereomer
  "rms_thresh": 1.0,          // RMSD threshold for conformer pruning
  "num_repr_confs": 3,        // Number of representative conformers to keep
  "e_rel_max": 25.0,          // Max relative energy (kcal/mol) for conformers
  "drop_close_energy": true,  // Drop conformers with very similar energies
  "only_global_minimum": false // If true, return ONLY the single lowest energy structure
}
```

**Parameters Detail:**

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `name` | `"complex"` | Name of the complex. |
| `res_structs` | `1` | Max resonance structures to consider for canonical ID generation. Increase for complex delocalized ligands. |
| `regime` | `"all"` | Stereomer search scope: `"CA"` (center only), `"ligands"` (ligands only), or `"all"`. |
| `get_enantiomers` | `False` | If `True`, keeps both enantiomers. If `False`, keeps only one. |
| `trans_cycle` | `False` | If `True`, allows trans-chelation (min cycle size 5). |
| `mer_rule` | `True` | If `True`, applies empiric rule to restrict impossible fac- geometries for rigid fragments. |
| `num_confs` | `10` | Number of initial conformers to generate per stereomer. |
| `rms_thresh` | `1.0` | RMSD threshold for pruning similar conformers. |
| `num_repr_confs` | `3` | Number of representative conformers to keep per stereomer. |
| `e_rel_max` | `25.0` | Max relative energy (kcal/mol) allowed for conformers. |
| `drop_close_energy` | `True` | If `True`, drops conformers with very similar energies. |
| `only_global_minimum` | `False` | If `True`, returns **only the single lowest energy structure** across all stereomers. |

**Response:**
```json
{
  "sdf": "... SDF content ...",
  "error": null,
  "logs": "..."
}
```
