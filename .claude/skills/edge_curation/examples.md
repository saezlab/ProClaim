# Worked Examples

Detailed case studies demonstrating the PPI curation workflow.

## Example 1: ALOX5 → STAT5A (Inhibition via phosphorylation)

**Claim**: ALOX5 inhibits STAT5A through phosphorylation

### Step 1: Parse
- Source: ALOX5
- Target: STAT5A
- Sign: Inhibition (downregulation)
- Mechanism: Phosphorylation

### Step 2: Characterise
- **ALOX5**: Arachidonate 5-lipoxygenase, a metabolic enzyme involved in leukotriene biosynthesis (pro-inflammatory lipid mediators)
- **STAT5A**: Signal transducer and activator of transcription 5A, a transcription factor involved in immune/inflammatory responses

### Step 3: Assess Plausibility
- **Red flag**: Metabolic enzyme claimed to directly phosphorylate a transcription factor
- Metabolic enzymes primarily catalyse metabolite conversions, not protein modifications
- However, ALOX5 products (leukotrienes) are pro-inflammatory, and STAT5 is involved in immune responses → plausible pathway connection

**Initial hypothesis**: Likely INDIRECT or FALSE for direct interaction

### Step 4: Search Evidence
- Search "ALOX5 STAT5A": No papers co-mentioning both in interaction context
- Search "ALOX5 STAT5": Found paper on ALOX5 inhibitor regulating tumour-associated macrophages via JAK-STAT
- Key finding: Elevated ALOX5 associated with **activation** of JAK-STAT pathway (opposite sign!)

### Step 5: Evaluate
- No direct evidence of ALOX5 phosphorylating STAT5A
- Pathway connection exists but:
  - Effect is indirect (through metabolites/signalling cascade)
  - Direction is opposite (activation, not inhibition)

### Verdict
```
VERDICT: FALSE

INTERACTION: ALOX5 → STAT5A (inhibition, phosphorylation)

EVIDENCE SUMMARY:
- No direct evidence of ALOX5-STAT5A physical interaction
- ALOX5 indirectly affects JAK-STAT pathway via pro-inflammatory metabolites
- Evidence suggests activation, not inhibition

NOTES:
- Likely pathway conflation: both proteins involved in inflammation
- ALOX5 produces pro-inflammatory leukotrienes that may activate JAK-STAT signalling
- Should be classified as indirect if any regulatory relationship exists
```

---

## Example 2: HCK → BCR (Activation via phosphorylation)

**Claim**: HCK activates BCR through phosphorylation

### Step 1: Parse
- Source: HCK
- Target: BCR
- Sign: Activation (upregulation)
- Mechanism: Phosphorylation

### Step 2: Characterise
- **HCK**: Hematopoietic cell kinase, a Src family tyrosine kinase
- **BCR**: Breakpoint cluster region protein; forms BCR-ABL fusion in certain leukaemias

### Step 3: Assess Plausibility
- **Favourable**: Kinase claimed to phosphorylate another protein
- Src family kinases commonly phosphorylate signalling proteins
- BCR-ABL is a well-studied oncogenic fusion

**Initial hypothesis**: Plausible, needs evidence verification

### Step 4: Search Evidence
- Search "HCK BCR": Multiple relevant papers found
- Paper 1 (EMBO J, 2002): HCK inhibition abrogates BCR-ABL activation
- Paper 2 (JBC, 1997): BCR-ABL preferentially binds inactive HCK; interaction stimulates HCK which may phosphorylate GRB2 binding site in BCR-ABL

### Step 5: Evaluate
- High-quality journals (EMBO J, JBC)
- Methods include immunoprecipitation, western blots, kinase assays
- **Complication**: Some evidence suggests BCR-ABL activates HCK (reverse direction)
- However, evidence also supports HCK phosphorylating BCR-ABL

### Verdict
```
VERDICT: TRUE

INTERACTION: HCK → BCR (activation, phosphorylation)

EVIDENCE SUMMARY:
- Multiple papers demonstrate HCK-BCR/BCR-ABL physical interaction
- Co-immunoprecipitation confirms binding
- Kinase activity studies support phosphorylation
- Note: Bidirectional regulation may exist

KEY REFERENCES:
- PMID from JBC 1997: HCK may phosphorylate GRB2 binding site in BCR-ABL
- PMID from EMBO J 2002: HCK inhibition abrogates BCR-ABL activation

NOTES:
- Original database may have flagged wrong phosphorylation site
- The edge HCK activates BCR is valid even if specific site was incorrect
- BCR vs BCR-ABL distinction may be relevant in some contexts
```

---

## Example 3: MAP2K1 → PPARG (Inhibition via phosphorylation)

**Claim**: MAP2K1 downregulates PPARG through phosphorylation

### Step 1: Parse
- Source: MAP2K1 (also known as MEK1)
- Target: PPARG (PPARγ, peroxisome proliferator-activated receptor gamma)
- Sign: Inhibition (downregulation)
- Mechanism: Phosphorylation

### Step 2: Characterise
- **MAP2K1**: Dual specificity mitogen-activated protein kinase kinase 1, part of ERK/MAPK pathway
- **PPARG**: Nuclear receptor transcription factor, regulates adipogenesis and metabolism

### Step 3: Assess Plausibility
- **Favourable**: Kinase claimed to phosphorylate transcription factor
- MAP2K1 is in a major signalling cascade known to regulate transcription factors
- Cross-talk between MAPK and nuclear receptor pathways is documented

**Initial hypothesis**: Plausible, check for direct evidence

### Step 4: Search Evidence
- Search "MAP2K1 PPARG": Multiple papers found
- Cell Cycle 2007: MAP2K1 acts as nucleocytoplasmic shuttle for PPARG
- Evidence shows ERK cascade attenuates PPARG transactivation via inhibitory phosphorylation

### Step 5: Evaluate
- Clear mechanistic evidence in peer-reviewed journal
- Mechanism explained: inhibitory phosphorylation modulates PPARG activity
- Direction and sign match the claim

### Verdict
```
VERDICT: TRUE

INTERACTION: MAP2K1 → PPARG (inhibition, phosphorylation)

EVIDENCE SUMMARY:
- Paper explicitly states ERK cascade (including MAP2K1) attenuates PPARG function
- Mechanism: inhibitory phosphorylation and nuclear export
- Robust experimental evidence

KEY REFERENCES:
- Cell Cycle 2007: MAP2K1/2 regulate PPARG via inhibitory phosphorylation

NOTES:
- Unclear why this was flagged as wrong in database update
- Evidence is explicit and methodology appears robust
```

---

## Example 4: CDK2 → UBE2R2 (Activation via phosphorylation)

**Claim**: CDK2 activates UBE2R2 through phosphorylation

### Step 1: Parse
- Source: CDK2
- Target: UBE2R2
- Sign: Activation (upregulation)
- Mechanism: Phosphorylation

### Step 2: Characterise
- **CDK2**: Cyclin-dependent kinase 2, cell cycle regulator
- **UBE2R2**: Ubiquitin-conjugating enzyme E2 R2, also known as CDC34B or **CK2** (!)

### Step 3: Assess Plausibility
- **Favourable**: Kinase claimed to phosphorylate another enzyme
- CDK2 has many substrates involved in cell cycle
- **Major red flag**: UBE2R2 alias "CK2" is very similar to "CDK2"

**Initial hypothesis**: Suspicious - possible name confusion artifact

### Step 4: Search Evidence
- Search "CDK2 UBE2R2": No papers co-mentioning in interaction context
- Search "CDK2 CDC34B": No relevant results
- Search "CDK2 CK2": Results confounded by name similarity
- NCBI Gene page confirms UBE2R2 has alias CK2

### Step 5: Evaluate
- No experimental evidence found supporting this interaction
- High probability of name confusion between CDK2 (kinase) and CK2 (UBE2R2 alias)
- Multiple aliases for UBE2R2 compound the search difficulty

### Verdict
```
VERDICT: FALSE

INTERACTION: CDK2 → UBE2R2 (activation, phosphorylation)

EVIDENCE SUMMARY:
- No papers found supporting direct interaction
- Exhaustive search including all known aliases yielded no results

NOTES:
- STRONG SUSPICION: UBE2R2 alias "CK2" likely confused with "CDK2"
- This is a classic name confusion artifact
- Database entry probably resulted from text-mining or annotation error

KEY REFERENCES:
- NCBI Gene entry for UBE2R2 (documents CK2 alias)
```

---

## Pattern Recognition Summary

| Pattern | Likely Outcome | Example |
|---------|---------------|---------|
| Metabolic enzyme → Transcription factor | FALSE or INDIRECT | ALOX5 → STAT5A |
| Kinase → Signalling protein | Check evidence carefully | HCK → BCR |
| Kinase → Transcription factor | Often TRUE | MAP2K1 → PPARG |
| Similar gene names involved | Suspect name confusion | CDK2 → UBE2R2 (CK2) |
| Both in same pathway, no direct evidence | Likely INDIRECT | Various |
| Evidence shows opposite sign | FALSE or WRONG SIGN | ALOX5 → STAT5A |
