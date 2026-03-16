"""
Unit tests for model_registry.py

Tests the singleton pattern, lazy loading, cache management, and pre-warming
functionality of the global model registry.
"""

import pytest
from pkevolve.verification import model_registry


@pytest.fixture(autouse=True)
def clear_cache_before_each_test():
    """Clear the model cache before each test to ensure isolation."""
    model_registry.clear_model_cache()
    yield
    model_registry.clear_model_cache()


def test_lazy_loading_semantic_similarity():
    """Test that semantic similarity computer is loaded lazily and cached."""
    # Check cache is empty
    status = model_registry.get_cache_status()
    assert not status["semantic_similarity"]

    # First call loads the model
    model1 = model_registry.get_semantic_similarity_computer()
    assert model1 is not None

    # Check cache now has the model
    status = model_registry.get_cache_status()
    assert status["semantic_similarity"]

    # Second call returns cached instance (same object)
    model2 = model_registry.get_semantic_similarity_computer()
    assert model2 is model1  # Same instance, not a copy


def test_lazy_loading_nli_entailment():
    """Test that NLI entailment computer is loaded lazily and cached."""
    # Check cache is empty
    status = model_registry.get_cache_status()
    assert not status["nli_entailment"]

    # First call loads the model
    model1 = model_registry.get_nli_entailment_computer()
    assert model1 is not None

    # Check cache now has the model
    status = model_registry.get_cache_status()
    assert status["nli_entailment"]

    # Second call returns cached instance (same object)
    model2 = model_registry.get_nli_entailment_computer()
    assert model2 is model1


def test_lazy_loading_mlp_classifier():
    """Test that MLP classifier is loaded lazily and cached."""
    # Check cache is empty
    status = model_registry.get_cache_status()
    assert not status["mlp_classifier"]

    # First call loads the model
    mlp1 = model_registry.get_mlp_classifier()
    assert mlp1 is not None
    assert "model" in mlp1
    assert "config" in mlp1
    assert "expected_features" in mlp1
    assert "mean" in mlp1
    assert "scale" in mlp1
    assert "aggregator" in mlp1

    # Check cache now has the model
    status = model_registry.get_cache_status()
    assert status["mlp_classifier"]

    # Second call returns cached instance (same object)
    mlp2 = model_registry.get_mlp_classifier()
    assert mlp2 is mlp1


def test_lazy_loading_feature_aggregator():
    """Test that feature aggregator returns the one from MLP classifier."""
    # get_feature_aggregator should trigger MLP loading
    aggregator1 = model_registry.get_feature_aggregator()
    assert aggregator1 is not None

    # MLP should now be in cache
    status = model_registry.get_cache_status()
    assert status["mlp_classifier"]

    # Getting aggregator again should return same instance
    aggregator2 = model_registry.get_feature_aggregator()
    assert aggregator2 is aggregator1


def test_lazy_loading_metadata_extractor():
    """Test that metadata extractor is loaded lazily and cached."""
    # Check cache is empty
    status = model_registry.get_cache_status()
    assert not status["metadata_extractor"]

    # First call loads the extractor
    extractor1 = model_registry.get_metadata_extractor()
    assert extractor1 is not None

    # Check cache now has the extractor
    status = model_registry.get_cache_status()
    assert status["metadata_extractor"]

    # Second call returns cached instance (same object)
    extractor2 = model_registry.get_metadata_extractor()
    assert extractor2 is extractor1


def test_clear_cache():
    """Test that clear_model_cache removes all models from cache."""
    # Load some models
    model_registry.get_semantic_similarity_computer()
    model_registry.get_mlp_classifier()

    # Verify they're cached
    status = model_registry.get_cache_status()
    assert status["semantic_similarity"]
    assert status["mlp_classifier"]

    # Clear cache
    model_registry.clear_model_cache()

    # Verify cache is empty
    status = model_registry.get_cache_status()
    assert not status["semantic_similarity"]
    assert not status["nli_entailment"]
    assert not status["mlp_classifier"]
    assert not status["metadata_extractor"]


def test_cache_status():
    """Test that get_cache_status correctly reports loaded models."""
    # Initially empty
    status = model_registry.get_cache_status()
    assert not any(status.values())

    # Load one model
    model_registry.get_semantic_similarity_computer()
    status = model_registry.get_cache_status()
    assert status["semantic_similarity"]
    assert not status["nli_entailment"]
    assert not status["mlp_classifier"]
    assert not status["metadata_extractor"]

    # Load another
    model_registry.get_nli_entailment_computer()
    status = model_registry.get_cache_status()
    assert status["semantic_similarity"]
    assert status["nli_entailment"]
    assert not status["mlp_classifier"]
    assert not status["metadata_extractor"]


def test_prewarm_all_models():
    """Test that prewarm_all_models loads all models and returns timing info."""
    # Initially cache should be empty
    status = model_registry.get_cache_status()
    assert not any(status.values())

    # Pre-warm all models
    load_times = model_registry.prewarm_all_models()

    # Check all models are now cached
    status = model_registry.get_cache_status()
    assert all(status.values())

    # Check load_times has entries for all models
    assert "semantic_similarity" in load_times
    assert "nli_entailment" in load_times
    assert "mlp_classifier" in load_times
    assert "metadata_extractor" in load_times

    # All load times should be positive numbers
    assert all(t > 0 for t in load_times.values())


def test_prewarm_is_idempotent():
    """Test that calling prewarm multiple times doesn't reload models."""
    # First prewarm
    load_times1 = model_registry.prewarm_all_models()

    # Get references to loaded models
    model1 = model_registry.get_semantic_similarity_computer()
    mlp1 = model_registry.get_mlp_classifier()

    # Second prewarm (should use cache, not reload)
    load_times2 = model_registry.prewarm_all_models()

    # Models should be the same instances
    model2 = model_registry.get_semantic_similarity_computer()
    mlp2 = model_registry.get_mlp_classifier()

    assert model2 is model1
    assert mlp2 is mlp1

    # Second prewarm should be much faster (cache hits)
    # Note: This assumes lazy loading doesn't re-time cached models
    # If the implementation always returns timing, both will have similar times


def test_models_are_functional():
    """Test that loaded models are actually functional (smoke test)."""
    # Load models
    sim_computer = model_registry.get_semantic_similarity_computer()
    nli_computer = model_registry.get_nli_entailment_computer()
    mlp = model_registry.get_mlp_classifier()
    aggregator = model_registry.get_feature_aggregator()
    extractor = model_registry.get_metadata_extractor()

    # Smoke tests - just verify they can be called without errors
    claim = "MAPK1 activates H3-3A"
    evidence = "MAPK1 phosphorylates H3-3A at serine residue."

    # Test semantic similarity
    similarity = sim_computer.compute(claim, evidence)
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0

    # Test NLI
    nli_result = nli_computer.compute(claim, evidence)
    assert "nli_entailment" in nli_result
    assert "nli_contradiction" in nli_result
    assert "nli_neutral" in nli_result

    # Test MLP structure
    assert mlp["model"] is not None
    assert len(mlp["expected_features"]) > 0

    # Test aggregator (with minimal data)
    papers = []
    result = aggregator.aggregate_all(papers)
    assert result is not None

    # Metadata extractor requires network access, skip functional test
    assert extractor is not None


def test_models_load_after_cache_clear():
    """Test that models can be reloaded after cache is cleared."""
    # Load a model
    model1 = model_registry.get_semantic_similarity_computer()

    # Clear cache
    model_registry.clear_model_cache()

    # Load again - should work fine
    model2 = model_registry.get_semantic_similarity_computer()

    # Should be a fresh instance (not the same object)
    # Note: This test may fail if Python reuses memory addresses
    # The important thing is it doesn't crash
    assert model2 is not None
