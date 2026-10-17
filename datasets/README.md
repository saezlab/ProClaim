# ProClaim-eval

Five claim subsets, 718 claims in total. Every claim carries a binary gold label, `SUPPORT` or `REFUTE`.

## Unpacking

The CSVs ship inside the password-protected archive `proclaim_eval.zip` (password: `proclaim-eval`)
so the claims and gold labels are not exposed as plain text to web crawlers. From the repository root:

```bash
bash scripts/unpack_datasets.sh
```

This writes the ten CSVs below into `datasets/` (`unzip -P proclaim-eval datasets/proclaim_eval.zip -d datasets`
does the same). The unpacked CSVs are git-ignored; please do not redistribute them in plain text.

| File | Subset | Domain | Claims | SUPPORT | REFUTE | Example claim |
|------|--------|--------|-------:|--------:|-------:|---------------|
| `signor.csv` | SIGNOR | molecular biology | 101 | 38 | 63 | GNAS directly activates ADCY1 |
| `connectomedb.csv` | ConnectomeDB | molecular biology | 305 | 184 | 121 | {Ligand} as ligand directly interacts extracellularly with {Receptor} as receptor |
| `extri.csv` | ExTRI | molecular biology | 125 | 49 | 76 | ATF2 directly activates transcription of MMP9 |
| `nubase.csv` | NUBASE | nuclear physics | 85 | 77 | 8 | Cobalt-52 has an isomeric state with a half-life of about 102 milliseconds |
| `nasa_exoplanet.csv` | NASA Exoplanet Archive | astronomy | 102 | 55 | 47 | EPIC 201257461.01 is a planet orbiting the star EPIC 201257461 |

## Columns

Shared by all subsets:

- `dataset`, `id`: subset name and claim identifier. In SIGNOR, `flip` marks the negated
  variant of a row, which shares the original's `id`; the code appends `_flip` to its claim id
  (the `_judge40` files already carry the suffix).
- `claim`: the claim text given to every system.
- `label`: gold label (`SUPPORT` / `REFUTE`).
- `original_label`: the label or status in the source database.
- `evidence`: the source record's reference (PMID, DOI or citation).

The remaining columns hold the structured fields the claim was built from (for example
`entity_a` / `entity_b` / `effect` for SIGNOR, `tf` / `target` for ExTRI, `nuclide` / `state` for NUBASE).

## 40-claim samples

`<subset>_judge40.csv` holds 40 claims per subset, drawn from the full CSV with the same columns.
They are the claims used for the judge comparison on all five subsets and for the ReAct run that
receives ProClaim's subclaims and extracted facts.
