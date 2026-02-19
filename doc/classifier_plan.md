# Scientific Claim Verification: Feature Engineering & Aggregation

This document defines the technical specifications for extracting features from individual papers (Local Features) and aggregating them across multiple papers (Global Aggregation). The objective is to construct a Hybrid Neuro-Symbolic system that combines NLP models, Metadata, and LLM reasoning capabilities for automated Claim Verification.

---

## Section 1: Local Feature Extraction (Single Paper Level)

In this phase, for each retrieved Evidence Paper $P_i$ and Claim $C$, we extract a feature vector $v_i$. These features are categorized into three groups: Metadata, NLP Metrics, and LLM Reasoning.

### 1. Metadata Features (Source Reliability & Context)
These features quantify the credibility and timeliness of the evidence source, acting as a Bayesian prior for the verification model.

* **Publication Year (Done)**
* **Journal Impact Factor (IF) (Done)**
    * **Definition:** The impact metric of the venue (journal/conference). Recommended preprocessing: $\log(1 + \text{IF})$.
* **Normalized Citation Count (Done)**
    * **Definition:** The citation count of the paper, normalized by its age.
    * **Formula:** $\frac{\text{Citation Count}}{\text{Current Year} - \text{Pub Year} + 1}$.
* **Author H-index (Max) (Done)**
    * **Definition:** The highest H-index among the paper's authors.
* **Document Type**
    * **Definition:** Categorical classification of the evidence level.
    * **Values:** `[Meta-analysis, RCT, Cohort Study, Case Report, Review, Opinion]`.
    * **Rationale:** Based on the Hierarchy of Evidence, a Meta-analysis or RCT carries significantly more weight than an Opinion piece or Case Report.
* **Section Source**
    * **Definition:** The specific section of the paper from which the evidence sentence was extracted.
    * **Values:** `[Abstract, Introduction, Methods, Results, Discussion]`.
    * **Rationale:** Evidence from `Results` is direct experimental data, whereas `Introduction` may cite external work, and `Discussion` is interpretative.

### 2. NLP Features (Semantics & Alignment)
These features capture the linguistic alignment and logical entailment between the Claim and the Evidence text.

* **Semantic Similarity**
    * **Definition:** Cosine similarity between the embeddings (e.g., SBERT) of the Claim and the Evidence.
    * **Rationale:** Measures topical proximity and relevance in the vector space.
* **Claim Entity Coverage (Done)**
    * **Definition:** Recall of Named Entities in the Claim that also appear in the Evidence: $\frac{|\text{Claim Entities} \cap \text{Evidence Entities}|}{|\text{Claim Entities}|}$.
    * **Rationale:** Since claims are much shorter than evidence texts (e.g., full abstracts), Jaccard similarity is diluted by the many entities in the evidence. Recall focuses on whether the evidence covers the specific entities mentioned in the claim, providing a more meaningful alignment signal.
* **Keyword Specificity (IDF-Weighted)**
    * **Definition:** The sum of IDF weights for overlapping tokens.
    * **Rationale:** Ensures the match is driven by specific, information-rich terms rather than common stopwords.
* **NLI (Natural Language Inference) Entailment Score**
    * **Definition:** Probabilities predicted by a fine-tuned Natural Language Inference model (e.g., DeBERTa-v3-large).
    * **Output:** Vector: $[P(\text{Entailment}), P(\text{Contradiction}), P(\text{Neutral})]$.
    * **Rationale:** The most direct semantic signal indicating whether the text supports or refutes the claim.

### 3. LLM-Native Features (Reasoning & Uncertainty)
Leveraging the zero-shot reasoning capabilities and intrinsic uncertainty metrics of Large Language Models.

* **LLM Verdict**
    * **Definition:** The classification label predicted by the LLM after reading the evidence (One-hot encoded).
    * **Values:** `[Supported, Refuted, Not Enough Info]`.
* **Verdict Confidence (Logprobs)**
    * **Definition:** The exponential of the log-probability of the generated verdict token.
    * **Formula:** $\exp(\text{LogProb}(\text{"Supported"}))$.
    * **Rationale:** Represents the model's intrinsic confidence (uncertainty) in its own judgment.
* **Context Sufficiency**
    * **Definition:** A score (Binary or 0-1) assessing whether the retrieved text contains *enough information* to verify the claim.
    * **Rationale:** Filters out paragraphs that are topically relevant but lack the specific evidentiary value needed.
* **Perplexity (PPL)**
    * **Definition:** The perplexity of generating the Claim conditioned on the Evidence: $P(\text{Claim} | \text{Evidence})$.
    * **Rationale:** If the Evidence supports the Claim, the Claim should follow naturally (low PPL). If they contradict, PPL is typically higher.

---

## Section 2: Feature Aggregation Methods (Multi-Paper Level)

Once we have collected Local Feature Vectors $\{v_1, v_2, ..., v_n\}$ for $N$ papers, we apply an Aggregation Layer to compress them into a single global vector ($V_{final}$) for the final MLP decision head.

### 1. Statistical Pooling
Captures the distribution characteristics of numerical features (NLI scores, Similarity, Logprobs).

* **Max Pooling (Best Evidence)**
    * Extracts the maximum Entailment Score and maximum Contradiction Score across $N$ papers.
    * *Rationale:* A claim's truth often depends on the "strongest piece of evidence" (e.g., one definitive RCT), not the average.
* **Mean / Median Pooling**
    * Calculates the average confidence and similarity across all retrieved papers.
    * *Rationale:* Measures the general strength of the evidence body.
* **Standard Deviation (Dispersion)**
    * Calculates the standard deviation of NLI scores.
    * *Rationale:* Measures heterogeneity. High deviation indicates conflicting evidence or mixed quality.

### 2. Voting & Consensus
Treats each paper as a vote to analyze the overall stance distribution.

* **Support / Refute Ratios**
    * Ratio of Supported verdicts: $\frac{N_{sup}}{N_{total}}$.
    * Ratio of Refuted verdicts: $\frac{N_{ref}}{N_{total}}$.
* **Net Consensus Score**
    * $Score = N_{sup} - N_{ref}$.
* **Entropy (Controversy Index)**
    * Shannon Entropy of the stance distribution: $H = - \sum p_i \log p_i$.
    * *Rationale:* High entropy indicates high controversy or disagreement. The model should likely predict `Uncertain`. Low entropy indicates strong consensus.

### 3. Weighted Aggregation
Recognizing that not all papers are equal, we weight NLP features using Metadata.

* **Impact-Weighted Scoring**
    * $\text{Score}_{weighted} = \sum_{i=1}^{n} \log(IF_i) \cdot \text{NLI\_Score}_i$.
    * *Rationale:* Gives higher voice to evidence from high-impact journals during aggregation.
* **Confidence-Weighted Scoring**
    * Uses LLM Logprobs to weight the verdicts.
    * *Rationale:* Down-weights evidence where the model itself is uncertain about the interpretation.

### 4. Temporal Dynamics
Considers the time dimension of evidence to detect shifts in scientific consensus.

* **Latest Stance**
    * Extracts the average stance of only the **Top-k most recent** papers.
* **Trend Slope**
    * Calculates the slope of NLI scores over time.
    * *Rationale:* Detects paradigm shifts (e.g., early studies suggested X, but recent consensus confirms Not X).