"""LLM reasoning module for gene regulation questions."""

import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from utils import get_interaction_prompt
from pathlib import Path


class StructuredOutcome(BaseModel):
    """Structured outcome from LLM reasoning."""
    reasoning: str = Field(description="LLM's reasoning")
    answer_text: str = Field(description="Raw text answer from LLM")
    answer: bool = Field(description="True/False answer")


def get_reasoning_prompt_components() -> tuple[str, str]:
    """Get the system message and examples for reasoning prompt.

    Returns:
        Tuple of (base_system, examples)
    """
    base_system = """You are a molecular biologist expert in biological interactions. Focus on evidence from 2018 onwards. Answer with ONLY a SINGLE Yes or No.
Provide your response in this EXACT format:
1. Evidence for YES: Analyze evidence and mechanisms that would support a positive answer
2. Evidence for NO: Analyze evidence and mechanisms that would support a negative answer
3. Final Assessment: Weigh the evidence from both sides and determine which is stronger
4. Answer: [Yes/No]
CRITICAL: The final line must be exactly "Answer: Yes" or "Answer: No" with no additional text after it.
"""

    examples = """Example 1:
Question: Does p53 up-regulate BAX?
Evidence for YES: p53 is a transcription factor that binds to specific DNA sequences. Studies have identified p53 binding sites in the BAX promoter region. During cellular stress, p53 accumulation leads to increased BAX mRNA and protein levels. ChIP-seq data confirms direct p53 occupancy at the BAX locus. This is a well-documented direct transcriptional activation.
Evidence for NO: Some cell types show BAX expression changes independent of p53 status. Other transcription factors like E2F1 can also activate BAX. p53 has many targets and BAX could be indirectly regulated through intermediate factors rather than direct binding.
Final Assessment: While other factors can influence BAX, the preponderance of evidence shows direct p53 binding to BAX promoter and transcriptional activation. Multiple independent studies with ChIP, reporter assays, and knockout experiments confirm this direct regulatory relationship. The alternative explanations don't negate the direct mechanism.
Answer: Yes

Example 2:
Question: Does insulin inhibit the activity of glucagon?
Evidence for YES: Insulin and glucagon have opposing physiological effects on blood glucose. When insulin levels are high, glucagon secretion decreases. Insulin signaling can suppress glucagon gene expression in pancreatic alpha cells. The two hormones create a coordinated metabolic response.
Evidence for NO: Insulin doesn't directly bind to or inhibit the glucagon protein itself. Insulin doesn't block glucagon from binding to its receptor. The molecular activity of glucagon (receptor binding, signal transduction) remains unchanged in the presence of insulin. They activate separate, opposing pathways rather than one directly inhibiting the other's molecular function.
Final Assessment: The question asks about "activity" which in molecular biology typically refers to the direct molecular function of a protein. While insulin and glucagon are physiologically antagonistic and insulin can suppress glucagon secretion, insulin does not directly inhibit glucagon's molecular activity (receptor binding and signaling). This is physiological antagonism, not direct molecular inhibition.
Answer: No

Example 3:
Question: Does TNF-alpha up-regulate apoptosis?
Evidence for YES: TNF-alpha binds to TNFR1 and recruits TRADD, FADD, and procaspase-8 to form the death-inducing signaling complex (DISC). This activates caspase-8, which initiates the caspase cascade leading to apoptosis. Numerous studies show TNF-alpha treatment increases apoptotic markers (cleaved caspases, PARP cleavage, DNA fragmentation) across many cell types.
Evidence for NO: TNF-alpha doesn't always cause apoptosis - in many contexts it promotes cell survival through NF-κB activation. The outcome depends on cellular context, concurrent signals, and levels of anti-apoptotic proteins like c-FLIP and IAPs. Some cells are completely resistant to TNF-alpha-induced apoptosis.
Final Assessment: While the outcome is context-dependent, the question asks whether TNF-alpha up-regulates (increases) apoptosis, not whether it exclusively causes apoptosis. The molecular mechanism for TNF-alpha promoting apoptosis through DISC formation and caspase activation is well-established. The fact that cells have mechanisms to block this doesn't negate that TNF-alpha has pro-apoptotic activity when those checkpoints are overcome.
Answer: Yes

Example 4:
Question: Does AMPK activate mTOR?
Evidence for YES: Both AMPK and mTOR are central metabolic regulators that respond to energy status. They're both involved in autophagy regulation and metabolic adaptation. Some studies show complex crosstalk between AMPK and mTOR pathways, and in certain contexts, their activities can be coordinated.
Evidence for NO: AMPK and mTOR have opposing roles in cellular metabolism. AMPK is activated during energy stress and promotes catabolic processes, while mTOR promotes anabolic processes during nutrient abundance. AMPK directly phosphorylates TSC2, which inhibits mTOR complex 1 (mTORC1). AMPK also directly phosphorylates Raptor, a component of mTORC1, leading to mTOR inhibition. This is a well-established inhibitory relationship.
Final Assessment: The molecular evidence overwhelmingly shows that AMPK inhibits rather than activates mTOR. The direct phosphorylation events (AMPK → TSC2 → mTOR inhibition, and AMPK → Raptor → mTOR inhibition) are well-characterized mechanisms. While the pathways interact in complex ways, the direct regulatory relationship is inhibitory, not activating.
Answer: No
"""

    return base_system, examples


def ask_gene_regulation_question(
    source_gene: str,
    target_gene: str,
    relationship: str,
    search_context: str = "",
    llm: ChatOpenAI | None = None,
    save_raw_response: bool = True,
    raw_response_path: str | None = None,
    repeat_idx: int | None = None
) -> StructuredOutcome:
    """Ask LLM about gene regulation relationship.

    Args:
        source_gene: Source gene name
        target_gene: Target gene name
        relationship: Relationship type (e.g., 'up-regulates')
        search_context: Optional context from web search
        llm: ChatOpenAI instance for reasoning
        save_raw_response: Whether to save raw LLM response
        raw_response_path: Path to save raw response (if None, uses current directory)
        repeat_idx: Optional repeat index for multiple runs of same edge

    Returns:
        StructuredOutcome with reasoning and answer
    """
    base_system, examples = get_reasoning_prompt_components()

    # Get natural language prompt based on relationship type
    interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)

    if search_context:
        query = f"""Scientific context: {search_context}

Does {interaction_prompt}?"""
    else:
        query = f"""Does {interaction_prompt}?"""

    prompt = base_system + examples + query

    response = llm.invoke(prompt)

    # Get the full response content
    # GLM-4.6 with reasoning-parser returns content in reasoning_content field
    full_content = response.content.strip() if response.content else ""

    # Check if there's reasoning_content in additional_kwargs (for GLM-4.6)
    if not full_content and 'reasoning_content' in response.additional_kwargs:
        full_content = response.additional_kwargs['reasoning_content'].strip()

    # If still empty, check response_metadata
    if not full_content and isinstance(response.response_metadata, dict):
        if 'reasoning_content' in response.response_metadata:
            full_content = response.response_metadata['reasoning_content'].strip()

    # Save the raw response for debugging
    if save_raw_response and raw_response_path:
        raw_path = Path(raw_response_path)
        raw_path.mkdir(parents=True, exist_ok=True)
        # Include repeat index in filename if provided
        if repeat_idx is not None:
            filename = f"raw_{source_gene}_{target_gene}_repeat{repeat_idx}.txt"
        else:
            filename = f"raw_{source_gene}_{target_gene}.txt"
        with open(raw_path / filename, 'w') as f:
            f.write(str(response))

    # Find the answer line and extract reasoning, answer_text, and answer_value
    marker = "Answer:"
    marker_pos = full_content.find(marker)

    # Default values when marker is not found
    reasoning = full_content
    answer_text = "MARKER_NOT_FOUND"
    answer_value = False

    if marker_pos >= 0:
        # Extract reasoning (everything before "Answer:")
        reasoning = full_content[:marker_pos].strip()

        # Calculate character position where to start looking for yes/no
        char_pos_before_answer = marker_pos + len(marker)
        # Look for the answer in the next 100 characters
        char_pos_after_answer = min(char_pos_before_answer + 100, len(full_content))

        # Extract the answer section including the marker
        answer_section = full_content[char_pos_before_answer:char_pos_after_answer].strip()

        # Extract just the first line for cleaner answer_text, including the marker
        answer_lines = answer_section.split('\n')
        if answer_lines:
            answer_text = marker + " " + answer_lines[0].strip()

        # Check if answer is "Yes" (case-insensitive)
        if 'yes' in answer_text.lower():
            answer_value = True
        else:
            answer_value = False
    else:
        # No marker found
        print("Warning: 'Answer:' marker not found in response")

    return StructuredOutcome(
        reasoning=reasoning,
        answer_text=answer_text,
        answer=answer_value
    )

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from utils import load_edges
    from tqdm import tqdm

    # Configuration
    # model = "gpt-oss-120b"
    model = "glm-4.6"
    llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="sk-dummy",
            model=model,
            temperature=1,
            max_tokens=4096,  # Increased for GLM-4.6 reasoning mode
        )
    single_test = False
    num_repeats = 10  # Number of times to repeat each edge
    print(f"Repeat number set to: {num_repeats}")
    
    if single_test:
        # Example usage
        source = "p53"
        target = "BAX"
        interaction = "up-regulates"
        result_path = Path(f"./results/reasoning/{model}/no_search/false_edges")

        result = ask_gene_regulation_question(
            source_gene=source,
            target_gene=target,
            relationship=interaction,
            llm=llm,
            save_raw_response=True,
            raw_response_path=str(result_path)
        )

        result_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / f"{source}_{target}.json", 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
    else:
        # Process both false edges and true edges
        edge_datasets = [
            {
                'name': 'false_edges',
                'csv_path': '../all_removed_edges_with_sources.csv',
                'result_subdir': 'false_edges'
            },
            {
                'name': 'true_edges',
                'csv_path': '../ground_truth_data/true_edges.csv',
                'result_subdir': 'true_edges'
            }
        ]

        for dataset in edge_datasets:
            print(f"\n{'='*60}")
            print(f"Processing {dataset['name']}")
            print(f"{'='*60}")

            # Load edges from CSV
            edges_df = load_edges(csv_path=dataset['csv_path'])
            print(f"Loaded {len(edges_df)} {dataset['name']}")

            # Create results directory
            result_path = Path(f"./results/reasoning/{model}/no_search/{dataset['result_subdir']}")
            result_path.mkdir(parents=True, exist_ok=True)

            # Process each edge with repeats
            total_iterations = len(edges_df) * num_repeats
            for idx, edge in tqdm(edges_df.iterrows(), total=len(edges_df), desc=dataset['name']):
                source_gene = edge['source_gene']
                target_gene = edge['target_gene']
                relationship = edge['relationship']

                # Repeat each edge num_repeats times
                for repeat_idx in range(num_repeats):
                    # Ask the question
                    result = ask_gene_regulation_question(
                        source_gene=source_gene,
                        target_gene=target_gene,
                        relationship=relationship,
                        llm=llm,
                        save_raw_response=True,
                        raw_response_path=str(result_path),
                        repeat_idx=repeat_idx
                    )

                    # Save structured result with repeat index in filename
                    filename = f"{source_gene}_{target_gene}_repeat{repeat_idx}.json"
                    with open(result_path / filename, 'w') as f:
                        json.dump(result.model_dump(), f, indent=2)

            print(f"\n{dataset['name']} processing complete! Results saved in: {result_path}")
            print(f"Total files: {len(edges_df)} edges × {num_repeats} repeats = {total_iterations} files")

        print(f"\n{'='*60}")
        print("All datasets processed successfully!")
        print(f"{'='*60}")
