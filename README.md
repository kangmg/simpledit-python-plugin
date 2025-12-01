# Simpledit Python Plugin

Python backend extensions for Simpledit.

## Requirements

- **Java 8+**: Required for `py2opsin` (IUPAC to SMILES conversion).

## Features

- **Opsin Integration**: Convert IUPAC names to SMILES using `py2opsin`.

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
