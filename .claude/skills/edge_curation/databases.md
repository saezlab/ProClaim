# Database and Search Reference

Quick reference for databases and search strategies during PPI curation.

## Protein Information Databases

### Gene/Protein Identity
| Database | Use Case | URL Pattern |
|----------|----------|-------------|
| NCBI Gene | Gene function, aliases, orthologs | ncbi.nlm.nih.gov/gene/?term=[SYMBOL] |
| UniProt | Protein function, domains, PTMs | uniprot.org/uniprotkb?query=[SYMBOL] |
| GeneCards | Comprehensive gene summaries | genecards.org/cgi-bin/carddisp.pl?gene=[SYMBOL] |

### Interaction Databases
| Database | Content | Notes |
|----------|---------|-------|
| STRING | Functional associations | Includes indirect/predicted; filter by evidence type |
| BioGRID | Curated physical interactions | High quality but not exhaustive |
| IntAct | Molecular interactions | Good for binding evidence |
| Signor | Signalling interactions | Signed, directed; source of many edges |
| KEGG | Pathway context | Useful for understanding indirect relationships |
| Reactome | Pathway reactions | Detailed mechanistic context |

## PubMed Search Strategies

### Basic Co-occurrence
```
[GENE1] AND [GENE2]
```

### Interaction-Specific
```
[GENE1] AND [GENE2] AND (phosphorylation OR binding OR interaction)
```

### Filter by Study Type
```
[GENE1] AND [GENE2] AND (immunoprecipitation OR "mass spectrometry" OR "kinase assay")
```

### Expand to Gene Family
```
[GENE1] AND [FAMILY_NAME] family
```

## Alias Lookup Strategy

1. Search NCBI Gene for the protein
2. Check "Also known as" field
3. Note all aliases including:
   - Official symbol
   - Previous symbols
   - Synonyms
   - Protein names

Example for UBE2R2:
- Official: UBE2R2
- Aliases: CDC34B, UBC3B, CK2

## Evidence Quality Tiers

### Tier 1 (Strong)
- Co-immunoprecipitation with both proteins detected
- In vitro kinase assay with purified proteins
- Mutagenesis confirming interaction site
- Structural data (X-ray, cryo-EM) showing complex

### Tier 2 (Moderate)
- Yeast two-hybrid
- Proximity ligation assay
- FRET/BRET
- Co-localisation with functional consequence

### Tier 3 (Weak/Indirect)
- Pathway co-membership
- Gene expression correlation
- Computational prediction
- Text-mining association

## Red Flag Checklist

- [ ] Gene names differ by only 1-2 characters
- [ ] One protein has many aliases
- [ ] No papers co-mention both proteins
- [ ] Interaction claim involves incompatible protein classes
- [ ] Evidence shows opposite direction or sign
- [ ] Only computational/predicted evidence available
- [ ] Proteins localise to different compartments
