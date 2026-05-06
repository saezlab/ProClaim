# Claim Verification Datasets

This supplementary package includes two claim-level evaluation datasets.

| File | Dataset | Rows |
|------|---------|------|
| `connectomedb.csv` | ConnectomeDB | 318 |
| `signor.csv` | SIGNOR | 101 |

## Labels

Both datasets use the three-way label set:

- `SUPPORT`
- `REFUTE`
- `UNCERTAIN`

### ConnectomeDB

| Label | Count |
|-------|-------|
| SUPPORT | 184 |
| REFUTE | 121 |
| UNCERTAIN | 13 |

Claims are ligand-receptor interaction statements of the form:
`{Ligand} as ligand directly interacts extracellularly with {Receptor} as receptor.`

### SIGNOR

| Label | Count |
|-------|-------|
| SUPPORT | 36 |
| REFUTE | 60 |
| UNCERTAIN | 5 |

Claims are direct interaction statements derived from SIGNOR source-target-effect rows, including flipped variants for supported examples.

## Optional Local Paths

`paths.yaml.template` can be copied to `paths.yaml` if you want to rebuild these CSVs from local source data. It is not required to run the bundled supplementary smoke tests.