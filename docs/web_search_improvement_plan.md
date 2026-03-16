# Web Search Improvement Plan: LLM-Generated Query Strategy

## Current State Analysis

### Phase 1: Entity-Based Initial Search (Current Default)
**Location**: [`evidence_api.py:234-259`](../src/pkevolve/verification/evidence_api.py#L234-L259) (`formulate_pubmed_query`) and [`evidence_api.py:372-423`](../src/pkevolve/verification/evidence_api.py#L372-L423) (`search_pubmed_progressive`)

**How it works**:
1. Extract entity names from claim using regex pattern: `r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b"`
   - Example: "Does MAPK1 activate H3-3A?" → extracts ["MAPK1", "H3-3A"]
2. Extract biological verbs and convert to nouns:
   - "activates" → "activation"
   - "phosphorylates" → "phosphorylation"
3. Generate tiered queries from strict to broad:
   - Tier 1: `(MAPK1 AND H3-3A) AND (activation)`
   - Tier 2: `(MAPK1 AND H3-3A)`
   - Tier 3: Bridge queries with family-level tokens

**Limitations**:
- ❌ Only works for protein-protein interaction (PPI) claims
- ❌ Regex pattern is too restrictive for general scientific claims
- ❌ Cannot handle diverse claim types:
  - Diagnosis claims: "JAK2 V617F is not associated with lymphoid leukemia (B-lineage ALL, T-ALL or CLL)"
  - Drug resistance claims: "NT5C2 K359Q mutation does not confer resistance to Nelarabine or Arabinosylguanine in T-cell ALL"
  - Disease mechanism claims
- ❌ Wastes time on entity checking as first step

### Phase 2: LLM-Generated Gap Queries (Already in RLM)
**Location**: [`subagents.py:454-514`](../src/pkevolve/verification/subagents.py#L454-L514) (`formulate_gap_queries`)

**When triggered**: After `check_sufficiency()` returns `INSUFFICIENT`

**How it works**:
1. **`identify_gaps(llm, claim, subclaims, facts)`** analyzes evidence gaps
   - Returns list of `Gap` objects with types (missing_subclaim_evidence, low_diversity, etc.)
2. **`formulate_gap_queries(llm, gaps)`** generates targeted PubMed queries from gaps
   - LLM creates 3-8 word queries specific to each gap type
   - Example output: `["MAPK1 phosphorylation mechanism", "H3 histone modification review"]`
3. **`search_for_gap(query, state)`** searches PubMed with these LLM-generated queries
   - Internally calls `search_pubmed()` with the LLM query

**Current workflow**:
```
Initial: search_pubmed_progressive(claim) → entity-based queries
    ↓
check_sufficiency(state, llm)
    ↓
if INSUFFICIENT:
    gaps = identify_gaps(llm, claim, subclaims, facts)
    queries = formulate_gap_queries(llm, gaps)  ← LLM generates queries!
    for query in queries:
        search_for_gap(query, state)  ← Search with LLM queries
    ↓
Repeat until sufficient or max iterations
```

**Key insight**: ✅ **LLM-generated queries ARE already used in RLM, but only for gap-filling searches**

### The Problem: Two-Phase Inefficiency

**Current reality**:
- Phase 1 (entity-based) ALWAYS runs first, even for non-PPI claims
- Phase 2 (LLM-based) only runs if Phase 1 fails to get sufficient evidence
- This creates a bottleneck for general scientific claims

**Why this is suboptimal**:
1. For PPI claims: Phase 1 works, but Phase 2 is often needed anyway
2. For general claims: Phase 1 often fails or returns irrelevant papers, wasting time

### Reference: paper_search_agent.py (NOT in RLM). This is an old version of implementation
**Location**: [`paper_search_agent.py:194-237`](../src/pkevolve/search/paper_search_agent.py#L194-L237) (`convert_question_to_query`)

**Status**: ⚠️ This is a **standalone tool**, not integrated into the RLM loop
- Used for one-off paper search tasks
- Could serve as inspiration for improving RLM's Phase 1 but since it is an old version of implementation, treat this only for reference. Use it if you think the implementation is logical and optimal.

### Reference: mcp-simple-pubmed Approach
**Location**: [`mcp_simple_pubmed/pubmed_search.py:41-152`](../mcp_simple_pubmed/pubmed_search.py#L41-L152)

**Key insights**:
- Direct query passthrough: accepts user/LLM-provided query string directly
- No entity extraction layer
- Supports full PubMed syntax (MeSH terms, field tags, Boolean operators)
- Simpler architecture: less preprocessing, more reliance on query quality

**Trade-offs**:
- ✅ Maximum flexibility
- ✅ Simpler code
- ❌ No automatic query broadening
- ❌ Requires LLM or user to formulate good queries

---

## Problem Statement

**Current bottleneck**: The two-phase approach forces entity extraction first, which:
1. **Fails for general claims**: Regex pattern `r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b"` only captures protein symbols
2. **Wastes a search iteration**: Must wait for Phase 1 to fail before using better LLM queries
3. **Inconsistent quality**: Phase 1 and Phase 2 use different query strategies

**Evidence of the problem**:
- Phase 2 (`formulate_gap_queries`) already shows LLM can generate good queries
- But we force every claim through entity extraction first
- For non-PPI claims, this means:
  - Iteration 1: Entity-based search (likely fails)
  - Iteration 2: LLM gap queries (works better)
  - Result: 2x iterations needed, slower convergence

**Goal**: Use LLM query generation as the **primary strategy from the start**, not just for gap-filling.

---

## Proposed Solution

### Strategy: Direct LLM Query Formulation

**Core idea**: Replace entity extraction with LLM-driven query generation. Initial search generates a single comprehensive query; gap handling provides iterative refinement when needed.

```
Workflow:
Claim → LLM Query Formulation → Single Search
    ↓ (if insufficient)
identify_gaps → formulate_gap_queries → search_for_gap
```

**Design principle**:
1. LLM generates a **single, comprehensive query** from the claim
2. If insufficient evidence, **gap handling** identifies what's missing and creates targeted follow-up queries
3. Single LLM-only path throughout - no entity extraction fallback

---

## Implementation Plan

### Phase 1: Create LLM Query Generator
**Goal**: LLM-driven query formulation for initial search

#### Step 1.1: Create `llm_query_generator.py`
- **File**: `src/pkevolve/search/llm_query_generator.py`
- **Key function**:
  ```python
  def generate_search_query(
      claim: str,
      llm: Callable,
  ) -> str:
      """
      LLM-driven query formulation for initial search.

      Args:
          claim: The scientific claim to verify
          llm: LLM callable

      Returns:
          PubMed query string
      """
  ```

#### Step 1.2: Query generation strategy

**Prompt template**:
```python
"You are a biomedical search expert. Generate a comprehensive PubMed query for:
Claim: {claim}

Requirements:
- Include key entities (genes, mutations, diseases, drugs)
- Add synonyms and common variants using OR operators
- Use appropriate field tags (e.g., [Title/Abstract], [MeSH])
- Keep query length reasonable (5-20 words)

Examples:
- Diagnosis claim: "JAK2 V617F is not associated with lymphoid leukemia"
  → (JAK2 AND V617F) AND (lymphoid leukemia OR B-ALL OR T-ALL OR CLL)

- Drug resistance claim: "NT5C2 K359Q mutation does not confer resistance to Nelarabine"
  → (NT5C2 AND K359Q) AND (Nelarabine OR Arabinosylguanine) AND (resistance OR sensitivity)

Output ONLY the query string. Use PubMed syntax."
```

#### Step 1.3: Query validation
```python
def _validate_query(query: str) -> bool:
    """Check for common PubMed syntax errors."""
    # Check for unbalanced parentheses
    if query.count("(") != query.count(")"):
        return False
    # Check for invalid operators
    if " AN " in query or " O " in query:  # Should be AND/OR
        return False
    # Warn if very short or very long
    word_count = len(query.split())
    if word_count < 2 or word_count > 30:
        return False
    return True

def _sanitize_query(query: str) -> str:
    """Fix common LLM query mistakes."""
    query = query.strip("`").strip()
    query = query.replace(" and ", " AND ").replace(" or ", " OR ")
    if "[" in query and "]" not in query:
        query = query.replace("[", "")
    return query
```

### Phase 2: Create `search_pubmed_llm`
**Goal**: Replace entity extraction with single LLM query

#### Step 2.1: Function signature
```python
def search_pubmed_llm(
    claim: str,
    state: EvidenceState,
    llm: Callable,
    max_results: int = 10,
) -> list[str]:
    """PubMed search using single LLM-generated query."""
```

#### Step 2.2: Implementation
```python
def search_pubmed_llm(claim, state, llm, max_results=10):
    """Single-shot search using LLM-generated query."""
    from pkevolve.search.llm_query_generator import generate_search_query

    # Generate comprehensive query
    query = generate_search_query(claim, llm)
    print(f"[LLM Query] {query}")

    # Search PubMed
    found, added, pmids = _search_and_add(query, state, max_results)
    print(f"  → Found {found}, added {added} new papers")

    if len(pmids) == 0:
        print("⚠️ No papers found in initial search. Gap handling will refine if needed.")

    return pmids
```

### Phase 3: Update Calling Sites
**Goal**: Replace all references to `search_pubmed_progressive` with `search_pubmed_llm`

#### Step 3.1: Update system prompts
- Update evidence_programming.py system prompt example
- Update repl_orchestrator.py workflow instructions
- Change from: `search_pubmed_progressive(state.claim, state)`
- Change to: `search_pubmed_progressive(state.claim, state, llm)`

#### Step 3.2: Update kernel namespace
- Update `kernel_runner.py` to inject the `llm` callable into kernel globals
- Ensure `llm` is available in notebook execution context

---

## Handling Edge Cases

### Q1: What if LLM search finds NO papers?

**Rely on gap-filling mechanism**:

```python
def search_pubmed_llm(claim, state, llm, max_results=10):
    query = generate_search_query(claim, llm)
    found, added, pmids = _search_and_add(query, state, max_results)

    if len(pmids) == 0:
        print("⚠️ No papers found in initial search.")
        # Return empty list - the RLM loop will detect INSUFFICIENT
        # and trigger gap-filling searches via formulate_gap_queries

    return pmids
```

**Workflow**:
1. Initial LLM query returns empty results
2. RLM's `check_sufficiency` detects INSUFFICIENT evidence
3. Gap handling (`identify_gaps` → `formulate_gap_queries`) generates refined queries
4. Gap queries provide iterative refinement as needed

### Q2: What if LLM generates invalid PubMed syntax?

**Validation & correction** (included in `llm_query_generator.py`):

```python
def generate_search_query(claim, llm, attempt):
    query = _call_llm_for_query(claim, llm, attempt)

    # Validate and sanitize
    if not _validate_query(query):
        print(f"⚠️ LLM generated invalid query: {query}")
        query = _sanitize_query(query)

    return query
```

See Phase 1 Step 1.3 for validation/sanitization implementation.


---

## Implementation Steps

### Step 1: Create LLM Query Generator
**New file**: `src/pkevolve/search/llm_query_generator.py`

**Functions to add**:
- [ ] `generate_search_query(claim: str, llm: Callable) -> str`
  - LLM-driven query formulation for initial search
  - Returns a single comprehensive PubMed query
- [ ] `_validate_query(query: str) -> bool`
  - Validate PubMed syntax (balanced parentheses, valid operators)
- [ ] `_sanitize_query(query: str) -> str`
  - Fix common LLM mistakes (case normalization, bracket matching)

**Testing**:
- [ ] Test with diverse claims (PPI, diagnosis, drug resistance)

### Step 2: Replace entity-based search with LLM-based search
**File**: `src/pkevolve/verification/evidence_api.py`

**Functions to remove**:
- [ ] `formulate_pubmed_query(claim: str) -> list[str]` (lines ~234-259)
  - Entity extraction using regex pattern
  - Biological verb to noun conversion
  - No longer needed with LLM query generation
- [ ] `_generate_tiered_queries(entities: list[str], verbs: list[str]) -> list[str]`
  - Tiered query generation (strict → broad)
  - Replaced by single LLM query
- [ ] `search_pubmed_progressive(claim, state, max_attempts, ...)` (lines ~372-423)
  - Old progressive search implementation
  - Name is misleading since we no longer do progressive search

**Functions to add**:
- [ ] `search_pubmed_llm(claim: str, state: EvidenceState, llm: Callable, max_results: int = 10) -> list[str]`
  - New function with clearer naming
  - Single LLM-generated query
  - Returns list of PMIDs added to state
  - Implementation:
    ```python
    def search_pubmed_llm(claim, state, llm, max_results=10):
        from pkevolve.search.llm_query_generator import generate_search_query
        query = generate_search_query(claim, llm)
        print(f"[LLM Query] {query}")
        found, added, pmids = _search_and_add(query, state, max_results)
        print(f"  → Found {found}, added {added} new papers")
        if len(pmids) == 0:
            print("⚠️ No papers found. Gap handling will refine if needed.")
        return pmids
    ```

### Step 3: Update Calling Sites
- [ ] **kernel_runner.py** (line ~216): Update namespace injection
  - Remove: `search_pubmed_progressive`
  - Add: `search_pubmed_llm`
  - Ensure `llm` is available in notebook globals

- [ ] **evidence_programming.py**: Update system prompts and imports
  - Line ~71: Replace `search_pubmed_progressive` with `search_pubmed_llm` in imports
  - Line ~127: Change instruction from `search_pubmed_progressive(state.claim, state)` to `search_pubmed_llm(state.claim, state, llm)`
  - Line ~182: Update workflow example

- [ ] **repl_orchestrator.py**: Update system prompts
  - Line ~70: Change instruction from `search_pubmed_progressive(state.claim, state)` to `search_pubmed_llm(state.claim, state, llm)`
  - Line ~95: Update documentation references

- [ ] **notebook_mcp.py** (line ~186): Update docstring references
  - Replace `search_pubmed_progressive` with `search_pubmed_llm`

- [ ] Update documentation files (doc/implementation_plan.md, doc/potential_extensions.md)

### Step 4: Testing and Validation
- [ ] Test on diverse claim types:
  - PPI claims (e.g., "MAPK1 activates H3-3A")
  - Diagnosis claims (e.g., "JAK2 V617F is not associated with lymphoid leukemia")
  - Drug resistance claims (e.g., "NT5C2 K359Q mutation does not confer resistance to Nelarabine")
- [ ] Measure: papers found, relevance, time to sufficiency
- [ ] Verify gap handling provides adequate refinement when initial query insufficient
- [ ] Ensure no regressions in existing test suite