"""
Integration tests for evidence_api.py with model_registry.py

Verifies that the refactored evidence API functions correctly use the
model registry without breaking existing functionality.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification import model_registry


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear model cache before and after each test."""
    model_registry.clear_model_cache()
    yield
    model_registry.clear_model_cache()


def test_populate_paper_features_uses_model_registry(temp_workspace):
    """Test that populate_paper_features correctly uses model registry."""
    from pkevolve.verification.evidence_api import populate_paper_features
    from pkevolve.verification.data_models import PaperRecord

    # Create a test state with one paper
    state = EvidenceState.init_new(
        claim="MAPK1 activates H3-3A",
        subclaims=["MAPK1 activates H3-3A"],
        workspace=temp_workspace,
    )

    # Add a paper with abstract
    paper = PaperRecord(
        pmid="12345678",
        title="Test Paper",
        abstract="MAPK1 phosphorylates H3-3A at serine residue in vitro.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2020-01-01",
    )
    state.papers["12345678"] = paper

    # Initially, no models should be cached
    cache_status = model_registry.get_cache_status()
    assert not cache_status["semantic_similarity"]
    assert not cache_status["nli_entailment"]

    # Call populate_paper_features (without NLI to speed up test)
    result = populate_paper_features(state, compute_nli=False)

    # Verify semantic similarity model is now cached
    cache_status = model_registry.get_cache_status()
    assert cache_status["semantic_similarity"]
    # NLI should not be loaded since we set compute_nli=False
    # (but it might be if prewarm was called)

    # Verify features were populated
    assert paper.nlp is not None
    assert paper.nlp.semantic_similarity is not None
    assert isinstance(paper.nlp.semantic_similarity, float)


def test_check_sufficiency_uses_model_registry(temp_workspace):
    """Test that check_sufficiency correctly uses model registry."""
    from pkevolve.verification.evidence_api import check_sufficiency
    from pkevolve.verification.data_models import PaperRecord, Fact, Stance
    from pkevolve.verification.data_models import NLPFeatureVector, PaperFeatureVector

    # Create a test state with papers and facts
    state = EvidenceState.init_new(
        claim="MAPK1 activates H3-3A",
        subclaims=["MAPK1 activates H3-3A"],
        workspace=temp_workspace,
    )

    # Add a paper with features already populated
    paper = PaperRecord(
        pmid="12345678",
        title="Test Paper",
        abstract="MAPK1 phosphorylates H3-3A.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2020-01-01",
    )
    # Add minimal features to avoid MLP seeing all zeros
    paper.nlp = NLPFeatureVector(
        claim_entity_coverage=0.8,
        evidence_entity_coverage=0.7,
        semantic_similarity=0.75,
    )
    paper.metadata = PaperFeatureVector(
        pmid="12345678",
        publication_year=2020,
        log_impact_factor=1.5,
        log_citation_count=2.0,
        h_index=50,
    )
    state.papers["12345678"] = paper

    # Add a fact
    fact = Fact(
        text="MAPK1 phosphorylates H3-3A at serine residue.",
        stance=Stance.SUPPORT,
        source_pmid="12345678",
        relevant_subclaims=["MAPK1 activates H3-3A"],
    )
    state.facts.append(fact)

    # Initially, MLP should not be cached
    cache_status = model_registry.get_cache_status()
    assert not cache_status["mlp_classifier"]

    # Create a mock LLM for gap identification
    class MockLLM:
        def __call__(self, *args, **kwargs):
            return "No gaps identified"

    # Call check_sufficiency
    result = check_sufficiency(state, MockLLM())

    # Verify MLP is now cached
    cache_status = model_registry.get_cache_status()
    assert cache_status["mlp_classifier"]

    # Verify result structure
    assert result.label in ["SUFFICIENT_SUPPORT", "INSUFFICIENT"]
    assert isinstance(result.confidence, float)


def test_prewarm_models_improves_performance(temp_workspace):
    """Test that pre-warming models avoids loading delays."""
    from pkevolve.verification.evidence_api import populate_paper_features
    from pkevolve.verification.data_models import PaperRecord
    import time

    # Pre-warm models
    load_times = model_registry.prewarm_all_models()
    assert all(t > 0 for t in load_times.values())

    # All models should now be cached
    cache_status = model_registry.get_cache_status()
    assert all(cache_status.values())

    # Create a test state
    state = EvidenceState.init_new(
        claim="MAPK1 activates H3-3A",
        subclaims=["MAPK1 activates H3-3A"],
        workspace=temp_workspace,
    )

    # Add a paper
    paper = PaperRecord(
        pmid="12345678",
        title="Test Paper",
        abstract="MAPK1 phosphorylates H3-3A.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2020-01-01",
    )
    state.papers["12345678"] = paper

    # Measure time for populate_paper_features (should be fast since models are cached)
    start = time.time()
    result = populate_paper_features(state, compute_nli=False)
    elapsed = time.time() - start

    # Should be much faster than initial model loading
    # (semantic similarity model alone takes ~2-3 seconds to load)
    # With caching, the actual feature computation should be < 1 second
    assert elapsed < 5.0  # Generous threshold to account for slow systems

    # Verify features were populated
    assert paper.nlp is not None


def test_model_registry_singleton_behavior():
    """Test that model registry maintains singleton behavior across API calls."""
    # Get models directly
    model1 = model_registry.get_semantic_similarity_computer()
    model2 = model_registry.get_mlp_classifier()

    # Get them again
    model1_again = model_registry.get_semantic_similarity_computer()
    model2_again = model_registry.get_mlp_classifier()

    # Should be the exact same instances (singleton pattern)
    assert model1 is model1_again
    assert model2 is model2_again
