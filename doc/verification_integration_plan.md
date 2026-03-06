# Verification Agent Integration Plan

**Goal**: Create a new verification agent script (`scripts/verification/demo_verification_with_features.py`) that integrates the feature extraction capabilities from `scripts/claude_sdk/extract_features_scifact.py` into the evidence programming loop.

## 1. Core Objectives

1.  **Switch to `mcp-simple-pubmed`**: Ensure the agent uses the `mcp-simple-pubmed` MCP server for retrieval (as used in `extract_features_scifact.py`) instead of the custom internal searcher.
2.  **Integrate Feature Extraction**: Port the metadata and NLP feature extraction logic to populate `PaperRecord`s with rich feature vectors.
3.  **Preserve Verification Loop**: Maintain the iterative "Search -> Extract -> Check Sufficiency" loop.

## 2. Component Architecture

### A. Data Models (`src/pkevolve/verification/data_models.py`)
We need to extend the `PaperRecord` model to store the extracted features.

*   **New Fields for `PaperRecord`**:
    *   `metadata`: `Optional[PaperFeatureVector]` (Year, Impact Factor, Citation Count, H-index)
    *   `nlp`: `Optional[NLPFeatureVector]` (Entity Overlap, Semantic Similarity)

### B. NLP Tools (`src/pkevolve/verification/nlp_tools.py`)
*New module to host the NLP logic ported from `extract_features_scifact.py`.*

1.  **`BiomedicalEntityExtractor`**:
    *   **Role**: Client for the `scispacy_server.py`.
    *   **Mechanism**: Uses `mcp.client.stdio` to communicate with the local server running in `src/servers/scispacy_server.py`.
    *   **Function**: `extract(text: str) -> list[str]`
2.  **`SemanticSimilarityComputer`**:
    *   **Role**: Computes cosine similarity between claim and evidence using `sentence_transformers`.

### C. Web Search Integration
*Switching retrieval backend.*

*   **Tool**: `mcp-simple-pubmed` (External MCP server).
*   **Integration**:
    *   The new agent will initialize an MCP Client connection to `mcp-simple-pubmed` (via `uvx` or similar).
    *   It will expose a function wrapper (e.g., `search_pubmed_tool`) that calls this MCP server and converts results into `PaperRecord` objects for the `EvidenceState`.

### D. Metadata Extraction
*Reuse existing components.*

*   **Tool**: `pkevolve.verification.feature_extractor.PaperFeatureExtractor`.
    *   Calculates Metadata features (H-index, IF, etc.).

## 3. New Agent Script (`scripts/verification/demo_verification_with_features.py`)

This script will adapt the logic of `demo_evidence_programming.py` but use the new tools.

### Workflow
1.  **Initialization**:
    *   Initialize `EvidenceState` & `LLMClient`.
    *   Initialize `PaperFeatureExtractor` & `SemanticSimilarityComputer`.
    *   **Connect to MCP Servers**:
        *   `scispacy-server` (for NLP features).
        *   `mcp-simple-pubmed` (for Search).

2.  **Search Loop (Interation N)**:
    *   **Search**: Call `mcp-simple-pubmed` to get PMIDs/Abstracts/Full texts.
        *   *Note*: This replaces the `mcp_tools.search_pubmed` call.
    *   **for new_pmid in papers**:
        *   **Feature Extraction**:
            *   `metadata = feature_extractor.extract_metadata(pmid)`
            *   `nlp = process_nlp_features(claim, text, entity_extractor, sim_computer)`
        *   **Store**: Save features into `PaperRecord`.
        *   **Log**: Display extracted features to the user. For the NLP features, it's not necessary to display the entities, just the overlap ratio and coverage.
        *   **LLM Fact Extraction**: Proceed with standard fact extraction.

3.  **Sufficiency & Verdict**:
    *   Run `check_sufficiency`.
    *   Emit final verdict.

## 4. Implementation Steps (Planned)

1.  **Refactor**: Create `src/pkevolve/verification/nlp_tools.py` with the ported NLP classes.
2.  **Update Config**: Add `BiomedicalEntityExtractor` settings.
3.  **Script Creation**: Write `demo_verification_with_features.py` implementing the workflow above.
4.  **Testing**: Run against sample claims to verify feature population.

## 5. Notes
*   **Classifier**: The `classifier_plan.md` mentions training a classifier. For this step, we are *ignoring* the training/inference of the full classifier. We are only implementing the **feature extraction pipeline** that will feed into it later.
*   **Environment**: Ensure the `scispacy_server.py` path is correctly resolved relative to the new script location.
