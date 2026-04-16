import json
import glob
import os
import argparse
# uv run python scripts/sufficiency_classifier/slm_llm_ablation/generate_sufficiency_prompts.py --base-dir results/signor_eval_20260330_205939


PROMPT_TEMPLATE = """You are a scientific evaluator. Given a CLAIM and a REFERENCE (paper titles, abstracts, and NLP features), assess whether the REFERENCE contains sufficient evidence to support or refute the CLAIM.

Assign a sufficiency score between 0.0 and 1.0:
- 1.0: The REFERENCE contains decisive evidence — either clearly supporting OR clearly refuting the CLAIM. Both directions are equally "sufficient". Papers that unanimously refute the claim deserve the same high score as papers that unanimously support it.
- 0.0: The REFERENCE is entirely irrelevant (entities, topic, and semantics are unrelated to the claim), or no papers were found.
- Values in between reflect partial relevance, inconclusive evidence, or genuinely conflicting evidence. Conflicting evidence means the papers are relevant and credible, but split in direction — some support the claim while others refute it — and after weighing their content, methodology, and credibility, no clear verdict can be reached. Do NOT judge conflict by the ratio of supporting vs. refuting papers alone; a single high-quality paper refuting the claim can outweigh several weaker ones supporting it, and vice versa.

Output the ### EXPLANATION (brief reasoning), then the ### EVALUATION (JSON).

### FEATURE DEFINITIONS
Each paper in the REFERENCE includes pre-computed Metadata and NLP fields:

Metadata:
- publication_year: Year the paper was published.
- log_impact_factor: log(1 + journal impact factor). Higher values indicate a more prestigious publication venue.
- normalized_citation_count: Citation count divided by paper age (citations / (current_year - pub_year + 1)). Measures age-adjusted community attention.
- author_h_index_max: The highest H-index among all authors. Serves as a proxy for author credibility.

NLP (computed relative to the CLAIM):
- claim_entity_coverage: Fraction of biomedical named entities in the claim that also appear in the paper (0–1). Measures how well the paper covers the specific entities mentioned in the claim.
- semantic_similarity: Cosine similarity between sentence-embedding vectors of the claim and the abstract (0–1). Measures topical proximity.
- nli_entailment: Probability (0–1) from a DeBERTa-v3-large NLI model that the nli_best_chunk_text ENTAILS (supports) the claim.
- nli_contradiction: Probability (0–1) that the nli_best_chunk_text CONTRADICTS (refutes) the claim.
- nli_neutral: Probability (0–1) that the nli_best_chunk_text is NEUTRAL (neither supports nor contradicts) the claim. Note: nli_entailment + nli_contradiction + nli_neutral ≈ 1.
- nli_best_chunk_text: The specific text passage from the paper that produced the highest NLI signal (i.e., most relevant chunk used to compute the NLI scores above).

### CLAIM
{claim}

### REFERENCE
{reference}
"""

def process_evidence_state(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    claim = data.get('claim', '')
    papers = data.get('papers', {}) or {}

    retained_papers_text = []

    for pmid, paper in papers.items():
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        metadata = paper.get('metadata') or {}
        nlp = paper.get('nlp') or {}

        # Filter nlp components that are too verbose or unnecessary
        nlp_filtered = {k: v for k, v in nlp.items() if k not in ('claim_entities', 'evidence_entities')}

        paper_text = (
            f"Paper {pmid}:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Metadata: {json.dumps(metadata)}\n"
            f"NLP: {json.dumps(nlp_filtered)}"
        )
        retained_papers_text.append(paper_text)

    if not retained_papers_text:
        reference_text = "No papers were found."
    else:
        reference_text = "\n\n".join(retained_papers_text)

    # Use str.replace instead of .format() to avoid conflicts with curly braces in JSON metadata
    return PROMPT_TEMPLATE.replace('{claim}', claim).replace('{reference}', reference_text)

def main():
    parser = argparse.ArgumentParser(description="Generate SLM/LLM prompts from evidence_state.json files.")
    parser.add_argument("--base-dir", type=str, default="results/signor_eval_20260330_205939",
                        help="Base directory to search for evidence states (read-only source)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory to mirror the source structure and write prompts. "
                             "Defaults to results/slm_llm_ablation/<base-dir-name>")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(base_dir), "slm_llm_ablation", os.path.basename(base_dir))

    pattern = os.path.join(base_dir, '**', 'evidence_state.json')
    json_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(json_files)} 'evidence_state.json' files.")
    print(f"Writing prompts to: {output_dir}")
    count = 0
    generated_files = []

    for file_path in json_files:
        try:
            prompt = process_evidence_state(file_path)

            # Mirror the relative path under output_dir
            rel_path = os.path.relpath(os.path.dirname(file_path), base_dir)
            dest_dir = os.path.join(output_dir, rel_path)
            os.makedirs(dest_dir, exist_ok=True)

            prompt_path = os.path.join(dest_dir, "llm_sufficiency_prompt.txt")
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            generated_files.append(prompt_path)
            count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully generated {count} prompts.")
    if count > 0:
        print("Example generated file:", generated_files[0])

if __name__ == "__main__":
    main()
