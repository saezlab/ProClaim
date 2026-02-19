
import unittest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.claude_sdk.extract_features_scifact import SemanticSimilarityComputer

class TestSemanticSimilarity(unittest.TestCase):
    def setUp(self):
        # Use the same model as in production
        self.computer = SemanticSimilarityComputer(model_name="all-MiniLM-L6-v2", chunk_size=20, chunk_overlap=5)

    def test_short_text_similarity(self):
        claim = "Gene A activates Gene B"
        evidence = "We found that Gene A directly activates Gene B in the experiment."
        score = self.computer.compute(claim, evidence)
        print(f"\nShort text score: {score}")
        self.assertGreater(score, 0.7)

    def test_contradiction_similarity(self):
        claim = "Gene A activates Gene B"
        evidence = "Gene A has no effect on Gene B."
        # SBERT similarity for contradiction might still be high-ish because of topic overlap, 
        # but usually lower than direct entailed text. 
        # However, for 'similarity' metric, we expect topic match to be high.
        # Let's just check it runs and gives a valid float.
        score = self.computer.compute(claim, evidence)
        print(f"Contradiction score: {score}")
        self.assertTrue(0.0 <= score <= 1.0)

    def test_chunking_logic(self):
        """Test that max pooling works over chunks."""
        claim = "The drug cures cancer."
        
        # Create a long text where the relevant part is at the end
        # chunk_size=20, so we need > 20 words generally
        # "noise" words
        noise = " ".join(["apple"] * 30) 
        target = "The drug completely cures cancer and saves lives."
        
        long_evidence = f"{noise} {noise} {target}"
        
        # If we didn't chunk, or if we averaged, the score would be diluted by 'apple'.
        # With max pooling, one chunk will contain the target and give high score.
        score = self.computer.compute(claim, long_evidence)
        print(f"Long text (max-pool) score: {score}")
        self.assertGreater(score, 0.6)

    def test_irrelevant_chunks(self):
        claim = "The drug cures cancer."
        long_evidence = " ".join(["apple"] * 100)
        score = self.computer.compute(claim, long_evidence)
        print(f"Irrelevant text score: {score}")
        self.assertLess(score, 0.5)

if __name__ == "__main__":
    unittest.main()
