# Feature Engineering Note: NLI Multi-Instance Aggregation

## Problem Statement
When calculating Natural Language Inference (NLI) features for scientific claims against long evidence documents, the standard approach is to split the document into manageable chunks (e.g., 256 tokens) due to model context limits (e.g., 512 tokens for `cross-encoder/nli-deberta-v3-large`).

However, aggregating these chunk-level NLI probabilities (Entailment, Contradiction, Neutral) into document-level features presents a challenge:
1. **Independent Max-Pooling (Raw Max):** Taking the max of each class independently `max(chunks, dim=0)` allows any single irrelevant chunk with a strong rhetorical stance (e.g., "The authors declare no role in the study") to dominate the `Contradiction` score, drowning out genuine `Entailment` signals from actual scientific findings in other chunks.
2. **Single Best Chunk:** Selecting a single "most opinionated" chunk and returning its exact Softmax distribution remains highly vulnerable to the exact same spurious correlation problem. A single high-confidence false positive will ruin the entire document's score.

## Solution: Semantic Similarity Weighting
To resolve this, we implemented a **Similarity-Weighted Max-Pooling** strategy.

### Methodology
1. **Compute Semantic Similarity:** Use a SentenceTransformer (`all-MiniLM-L6-v2`) to compute the Cosine Similarity between the Claim and *each* Evidence Chunk. This yields a similarity scalar $S_i \in [-1, 1]$ for each chunk $i$.
2. **Compute NLI Probabilities:** Use the CrossEncoder to compute the Softmax NLI probability distribution $P_i = [P_{ent}, P_{con}, P_{neu}]$ for each chunk $i$.
3. **Weighting:** Multiply the NLI probabilities by the semantic similarity: $W_i = P_i \times S_i$
4. **Selection:** Find the single chunk $j$ that maximizes the absolute opinionated weighted score: $j = \argmax_i( \max(W_{i,ent}, W_{i,con}) )$
5. **Extraction:** Return the *unweighted* probabilities $[P_{j,ent}, P_{j,con}, P_{j,neu}]$ and the text of the single best chunk $j$.

### Why this works
This approach elegantly filters out spurious NLI signals while ensuring that the final output probabilities are a coherent, true Softmax distribution (summing to 1) that represents a real single paragraph's stance.

An irrelevant acknowledgement section (e.g. "no role in this study") might yield a near 1.0 Contradiction probability from the NLI model, but its Semantic Similarity to the scientific claim will be very low (e.g. 0.3). Therefore, its weighted score ($1.0 \times 0.3 = 0.3$) will be outranked by the genuine supporting paragraph that has both a high NLI probability *and* a high Semantic Similarity (e.g. $W_{ent} = 0.8 \times 0.7 = 0.56$).

By weighting the NLI probabilities before selection, we ensure that the chosen `best_chunk` is actually topically relevant to the claim, creating much more robust features for downstream classifiers while preserving human-interpretable single-evidence extraction.
