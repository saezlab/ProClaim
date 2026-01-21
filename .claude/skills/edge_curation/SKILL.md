---
name: ppi-curation
description: Curate and validate signed directed protein-protein interactions (PPIs) from databases. Use when asked to verify if a protein interaction is real or a database artifact, validate edges like "A activates/inhibits B", determine if an annotated interaction is direct or indirect, or assess the quality of PPI database entries. Supports phosphorylation, binding, and regulatory interactions.
---

# Signed Directed PPI Curation

Validate whether a given protein-protein interaction (e.g., "ALOX5 inhibits STAT5A via phosphorylation") represents a true biological relationship or a database artifact.

## Workflow Overview

1. Parse the interaction claim
2. Characterise both proteins
3. **Check for confusable names** ← NEW STEP
4. Assess biological plausibility
5. Search for evidence
6. Evaluate evidence quality
7. Render verdict with justification

## Step 1: Parse the Interaction Claim

Extract and confirm:
- **Source protein** (regulator): The protein exerting the effect
- **Target protein** (regulated): The protein being affected
- **Sign**: Activation (upregulation) or inhibition (downregulation)
- **Mechanism** (if specified): Phosphorylation, binding, catalysis, etc.

Example: "HCK upregulates BCR through phosphorylation"
→ Source: HCK | Target: BCR | Sign: activation | Mechanism: phosphorylation

## Step 2: Characterise Both Proteins

For each protein, determine:
- **Functional class**: Kinase, phosphatase, transcription factor, metabolic enzyme, receptor, scaffold, etc.
- **Known aliases**: Critical for literature search (e.g., UBE2R2 is also called CDC34B or CK2)
- **Gene family membership**: STAT5A belongs to STAT family; consider other family members in searches

Use NCBI Gene, UniProt, or GeneCards. Record aliases for search expansion.

## Step 3: Check for Confusable Names (CRITICAL)

**Before searching for evidence**, actively check if the source or target protein name could be confused with a lexically similar but functionally different protein.

### High-Risk Confusable Patterns

| Pattern | Example | Why It's Dangerous |
|---------|---------|-------------------|
| Similar abbreviations | CDK2 vs CK2 | Different kinase families |
| Number transposition | STAT5A vs STAT3 | Different family members |
| Prefix/suffix confusion | ERK vs ERK2 vs ERK1/2 | Ambiguous specificity |
| Kinase family confusion | CDK vs CK vs CSNK | Easy to mis-type or conflate |
| Name embedding | ProteinX vs ProteinX-Phosphatase | One name contains the other conceptually |

### Known Confusable Pairs (Non-Exhaustive)

| Protein A | Protein B | Notes |
|-----------|-----------|-------|
| **CDK2** (cyclin-dependent kinase) | **CK2** (casein kinase 2) | Different families, similar abbreviations |
| **UBE2R2** (E2 ligase) | **CK2** (alias confusion) | UBE2R2 alias is CDC34B |
| **PP1** (phosphatase) | **PPP1** (same thing) | Nomenclature variants |
| **AKT1** | **ATK1** (typo) | Transcription errors |
| **RAF1** | **RAP1** | Similar abbreviations |
| **MEK1** | **MEKK1** | Different cascade levels |
| **[Kinase]** | **[Kinase] phosphatase** | Enzyme-regulator pairs with embedded names |

### Action: Confusable Name Search

If the source protein abbreviation is lexically similar to another protein (especially one with different function or substrate specificity):

1. **Identify the confusable protein** (e.g., CDK2 → could be confused with CK2)
2. **Search for [CONFUSABLE] + [TARGET]** in addition to the original search
3. **If the confusable search yields hits but original doesn't** → likely annotation artifact

Example workflow:
```
Claim: CDK2 phosphorylates TargetX
Step 1: Search "CDK2 TargetX" → sparse results
Step 2: Identify confusable: CK2 (Casein Kinase 2, different family)
Step 3: Search "CK2 TargetX" → strong hits about CK2 phosphorylation
Conclusion: Likely CDK2/CK2 name confusion artifact
```

### Red Flags for Name Confusion

- Evidence found uses a **slightly different abbreviation** of the gene name (CK2 vs CDK2)
- Found papers describe a **different kinase family** or **different substrate specificity**
- The claimed mechanism doesn't match the source protein's known targets
- Source protein's canonical substrates don't include the target protein class

## Step 4: Assess Biological Plausibility

Based on protein classes, estimate prior likelihood:

| Source Class | Likely Direct Interactor? | Notes |
|--------------|--------------------------|-------|
| Kinase | High | Primary function is phosphorylating other proteins |
| Phosphatase | High | Directly dephosphorylates targets |
| Transcription factor | Medium | Usually regulates genes, but can have protein partners |
| Metabolic enzyme | Low | Primary function is metabolite conversion; protein interactions are secondary |
| Scaffold/adaptor | High | Function depends on protein-protein binding |
| Receptor | Medium | Often requires ligand; check context |

**Red flags for false positives:**
- Metabolic enzyme claimed to directly regulate a transcription factor
- No obvious mechanistic link between protein functions
- Proteins operate in entirely different cellular compartments
- **Source protein functional class doesn't match claimed mechanism** (e.g., kinase claimed to "activate" without phosphorylation)
- **Claimed sign conflicts with source protein's canonical function** (e.g., phosphatase "activating" via phosphorylation)

**Consider indirect mechanisms:**
- Metabolic enzymes may produce metabolites that allosterically regulate targets
- Pathway crosstalk (e.g., JAK-STAT pathway activation by upstream signals)
- These are often mis-annotated as direct interactions

## Step 5: Search for Evidence

### 5.1 Initial Search

Search for both proteins together:
```
[GENE1] [GENE2]
```

Examine first page of results. Check if:
- Both genes appear in the same abstract
- There is any mention of physical or functional interaction
- The interaction direction/sign matches the claim

### 5.2 Targeted Searches

If initial search is inconclusive, try:

```
[GENE1] phosphorylates [GENE2]        # For phosphorylation claims
[GENE1] activates [GENE2]             # For activation claims
[GENE1] inhibits [GENE2]              # For inhibition claims
[GENE1] [GENE2] interaction           # General interaction
[GENE1] [GENE2] binding               # Physical binding
```

### 5.3 Expand with Aliases and Family Members

If no results, search using:
- Known aliases (e.g., search "CDK2 CDC34B" instead of "CDK2 UBE2R2")
- Family name (e.g., "ALOX5 STAT" instead of "ALOX5 STAT5A")
- Parent gene (e.g., "ALOX5 JAK-STAT pathway")

### 5.4 Confusable Name Search (Critical When Evidence Is Sparse)

**If initial searches yield few or no results:**

1. Return to Step 3 confusable names list
2. Search for `[CONFUSABLE_NAME] [TARGET]`
3. If this search yields strong hits, the annotation is likely an artifact

Example:
```
Original: "CDK2 TargetX" → 0 direct results
Confusable: "CK2 TargetX" → Multiple papers on CK2 phosphorylating TargetX
Conclusion: Database likely confused CDK2 with CK2
```

### 5.5 Use PubMed for Rigorous Search

For definitive validation, use PubMed:
- Search both gene symbols
- Filter by publication type (experimental studies preferred)
- Check for co-occurrence in abstracts

### 5.6 Examine ALL Retrieved Papers for Confusion Sources (CRITICAL)

When searching for evidence, you will often retrieve papers that mention both proteins but do NOT support a direct interaction. **Do not dismiss these papers.**

**Critical question for EVERY retrieved paper:**
> "Could this paper be the source of a mis-annotation?"

Papers that mention both proteins without showing direct interaction may reveal:
- A **third protein that regulates BOTH** the source and target (shared regulator artifact)
- **Naming similarities** that could cause text-mining errors (e.g., a protein whose name contains part of the source protein's name)
- **Pathway co-membership** misinterpreted as direct interaction
- A **functional opposite** (e.g., a phosphatase for a kinase) that acts on both proteins

**Action:** When evidence for the claimed edge is sparse or absent, systematically re-examine ALL retrieved papers and ask:

1. Does this paper mention a protein with a similar name to the source?
2. Does this paper describe a protein that acts on BOTH the claimed source and target?
3. Could the relationship described in this paper have been misinterpreted as the claimed edge?

If you find a paper where both proteins appear in the context of a shared regulator, naming similarity, or pathway relationship — this likely explains the spurious annotation, even if the paper itself does not support the claimed direct interaction.

## Step 6: Evaluate Evidence Quality

### Paper Quality Indicators

| Factor | Strong Evidence | Weak Evidence |
|--------|----------------|---------------|
| Journal | High-impact, peer-reviewed (Nature, Cell, EMBO J, JBC) | Predatory journals, preprints without validation |
| Methodology | Co-immunoprecipitation, mass spec, kinase assays, mutagenesis | Correlation only, computational prediction |
| Directness | Shows physical binding or enzymatic activity | Pathway association, expression correlation |
| Specificity | Tests specific proteins named in claim | Tests family members or related proteins |

### Key Experimental Methods Supporting Direct Interaction

- **Co-immunoprecipitation (Co-IP)**: Proteins physically associate
- **In vitro kinase assay**: Source kinase phosphorylates target
- **Yeast two-hybrid**: Physical binding detected
- **FRET/BRET**: Proteins in close proximity in cells
- **Mass spectrometry**: Identifies phosphorylation sites

### Direction/Sign Verification

Ensure the evidence supports the **claimed direction**:
- If claim is "A activates B", evidence should show A increases B activity/expression
- If evidence shows opposite direction (B activates A), mark as **likely false** or **reversed**
- If evidence shows opposite sign (activation vs claimed inhibition), mark as **wrong sign**

### Name Verification in Found Papers

When reviewing papers, explicitly verify:
- The paper uses the **exact gene symbol** from the claim (not a confusable)
- The protein described has the **same functional class** as claimed
- Any abbreviations in the paper are **unambiguous**

## Step 7: Render Verdict

### Verdict Categories

**TRUE**: Direct evidence supports the interaction
- Paper(s) demonstrate physical interaction or direct enzymatic relationship
- Direction and sign match the claim
- Methodology is appropriate for the claimed mechanism

**FALSE**: Evidence contradicts or does not support the interaction
- No papers co-mention both proteins in interaction context
- Evidence shows opposite direction or sign
- **Likely name confusion or pathway conflation** ← Flag this explicitly

**FALSE (NAME CONFUSION)**: Specific sub-category
- Evidence exists for a confusable protein name
- The confusable protein has opposite function or similar naming
- Database annotation is an artifact of text-mining or manual error

**FALSE (SHARED REGULATOR)**: Specific sub-category
- Both proteins are regulated by a common third protein
- Co-occurrence in literature is due to shared regulator, not direct interaction
- The shared regulator's name may be similar to one of the claimed proteins

**INDIRECT**: Interaction exists but is not direct
- Proteins are in same pathway but separated by intermediates
- Metabolite-mediated regulation
- Transcriptional regulation (source affects target gene expression, not protein)

**UNCERTAIN**: Insufficient evidence to determine
- Conflicting reports
- Only computational predictions available
- Evidence is tangential or low quality

### Output Format

```
VERDICT: [TRUE/FALSE/FALSE (NAME CONFUSION)/FALSE (SHARED REGULATOR)/INDIRECT/UNCERTAIN]

INTERACTION: [Source] → [Target] ([Sign], [Mechanism])

EVIDENCE SUMMARY:
- [Brief description of supporting or contradicting evidence]

NAME CONFUSION CHECK:
- Confusable proteins identified: [List any]
- Evidence for confusable: [Yes/No - brief description]

ARTIFACT SOURCE CHECK:
- Papers mentioning both proteins: [List]
- Likely source of mis-annotation: [Describe if identified]

KEY REFERENCES:
- [PMID or DOI]: [One-line summary of finding]

NOTES:
- [Any caveats, alternative interpretations, or flags]
```

## Common Pitfalls

### Name Confusion (HIGH PRIORITY)

**Similar Abbreviations, Different Families:**
- CDK2 (cyclin-dependent kinase) ↔ CK2 (casein kinase 2)
- UBE2R2 (alias: CDC34B) confused with other proteins
- Different isoforms (STAT5A vs STAT5B) conflated
- Fusion proteins (BCR-ABL) vs individual components (BCR, ABL)

**Embedded Name Confusion:**
- A protein name that contains another protein's name or abbreviation
- Example: "[X] Kinase" vs "[X] Kinase Phosphatase" — the phosphatase name embeds the kinase name
- Text-mining may conflate these due to shared terminology

**Text Mining Artifacts:**
- Similar abbreviations captured as same entity
- Gene symbols appearing in unrelated contexts
- Organism-specific variants conflated

### Shared Regulator Artifacts (HIGH PRIORITY)

When a third protein acts on BOTH the claimed source and target, their co-occurrence in literature can be misannotated as a direct interaction.

**Pattern to check:**
- Is there a regulator (kinase, phosphatase, E3 ligase, etc.) that modifies BOTH proteins?
- Do they appear together in papers about this shared regulator?
- Does the shared regulator have a name similar to either protein?

**High-risk scenarios:**
- Signaling protein (kinase, MAPK) as claimed source + chromatin protein (histone, TF) as target
- Both proteins are substrates of the same phosphatase or kinase
- The regulator's name contains abbreviations similar to the claimed source

### Pathway Conflation
- Proteins in same pathway annotated as directly interacting
- Downstream effects attributed as direct interactions
- Shared upstream regulator creates apparent correlation

### Annotation Artifacts
- Correct interaction but wrong phosphorylation site
- Correct interaction but wrong direction
- Mouse/human ortholog confusion

### Publication Bias
- Older papers may have less rigorous methods
- Review articles may propagate errors from primary sources
- Computational predictions cited as experimental evidence

## Quick Decision Tree

```
1. Is there a confusable protein name? (CHECK FIRST!)
   YES → Search for confusable + target
         If confusable has evidence but original doesn't → FALSE (NAME CONFUSION)
   NO → Continue

2. Can source protein plausibly regulate target directly?
   NO → Likely FALSE or INDIRECT
   YES → Continue

3. Do papers co-mention both proteins?
   NO → Likely FALSE (search with aliases first)
   YES → Continue

4. Do papers show direct interaction evidence?
   NO → Check Step 5.6: Could these papers explain a mis-annotation?
        (shared regulator, naming similarity, pathway co-membership)
        If yes → FALSE (NAME CONFUSION) or FALSE (SHARED REGULATOR)
        If no → Check if INDIRECT
   YES → Continue

5. Does evidence match claimed direction and sign?
   NO → FALSE or WRONG SIGN
   YES → TRUE
```

## Appendix: Expanding the Confusable Names List

When you encounter a new name confusion case, document it:

```
| Protein A | Protein B | Confusion Type | How Discovered |
|-----------|-----------|----------------|----------------|
| CDK2 | CK2 | Different kinase families | Example edge validation |
| [Kinase X] | [Kinase X] Phosphatase | Embedded name | Shared regulator artifact |
```

This helps build institutional knowledge and prevents repeated errors.
