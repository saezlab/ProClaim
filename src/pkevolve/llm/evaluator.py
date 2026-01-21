"""
LLM Evaluator module for gene regulation questions.
Refactored and generalized from langgraph/llm_reasoning_fixed.py.
"""

from typing import Optional
import json
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from pkevolve.utils.signor_utils import construct_signor_question

class StructuredOutcome(BaseModel):
    """Structured outcome from LLM reasoning."""
    reasoning: str = Field(description="LLM's reasoning")
    answer_text: str = Field(description="Raw text answer from LLM")
    answer: Optional[bool] = Field(description="True/False answer, or None if uncertain")
    reasoning_content: Optional[str] = Field(default=None, description="Reasoning content if available (e.g., from reasoning models)")
    usage: Optional[dict] = Field(default=None, description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)")

class GeneInteractionEvaluator:
    """Evaluator for gene interaction questions using LLM."""

    def __init__(
        self, 
        client: OpenAI, 
        model: str = "gpt-oss-120b",
        temperature: float = 1.0
    ):
        """
        Initialize the evaluator.
        
        Args:
            client: OpenAI client instance
            model: Model identifier
            temperature: Sampling temperature
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        
        self.base_system_prompt = """You are a molecular biologist expert in biological interactions. Answer with ONLY a SINGLE word: Yes, No, or None.
Provide your response in this EXACT format:
1. Evidence for YES: Analyze evidence and mechanisms that would support a positive answer
2. Evidence for NO: Analyze evidence and mechanisms that would support a negative answer
3. Final Assessment: Weigh the evidence from both sides and determine which is stronger. If the evidence is insufficient, contradictory, or you are uncertain, you should answer None.
4. Answer: [Yes/No/None]

IMPORTANT: Use "None" when:
- The evidence is insufficient to make a confident determination
- The evidence is contradictory or equally balanced
- The scientific literature does not provide clear consensus
- You are uncertain and essentially saying "I don't know"

CRITICAL: The final line must be exactly "Answer: Yes" or "Answer: No" or "Answer: None" with no additional text after it.
"""
        self.examples = """Example 1:
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
Question: Does TNF-alpha up-regulate NF-kB?
Evidence for YES: TNF-alpha is a well-known inflammatory cytokine that activates signaling cascades. When TNF-alpha binds to its receptor TNFR1, it triggers downstream signaling. NF-kB is a transcription factor involved in inflammation and immune responses. Many studies show NF-kB activation following TNF-alpha treatment.
Evidence for NO: NF-kB is not directly up-regulated by TNF-alpha in the sense of increased NF-kB expression. Instead, TNF-alpha causes the degradation of IkB (the inhibitor of NF-kB), which releases pre-existing NF-kB to translocate to the nucleus. The question uses "up-regulate" which typically means increasing gene expression or protein levels, not activation of existing protein.
Final Assessment: This question is ambiguous and depends on interpretation. If "up-regulate" means transcriptional increase of NF-kB itself, the answer would be No. If it means activation/nuclear translocation of NF-kB, the answer might be interpreted as Yes. The scientific literature uses varied terminology for this relationship. Without clear definition of "up-regulate" in this context, I cannot confidently determine the intended answer.
Answer: None
"""

    def construct_prompt(
        self,
        source_gene: str,
        target_gene: str,
        relationship: str,
        search_context: str = ""
    ) -> str:
        """
        Construct the full prompt for the LLM.
        
        Args:
            source_gene: Source entity
            target_gene: Target entity
            relationship: Interaction type
            search_context: Optional context from search
            
        Returns:
            Full prompt string
        """
        # Use existing utility to construct the question
        question = construct_signor_question(source_gene, target_gene, relationship)
        
        if search_context:
            query = f"""Scientific context: {search_context}\n\n Based on this context, generate a short summary of the context, then answer the question: {question}"""
        else:
            query = f"""{question}"""
            
        prompt = self.base_system_prompt + "\n" + self.examples + "\n" + query
        return prompt

    def evaluate(
        self,
        source_gene: str,
        target_gene: str,
        relationship: str,
        search_context: str = "",
        save_raw_path: Optional[Path] = None
    ) -> StructuredOutcome:
        """
        Evaluate if the relationship is true based on LLM reasoning.
        
        Args:
            source_gene: Source entity
            target_gene: Target entity
            relationship: Interaction type
            search_context: Optional context from search
            save_raw_path: Optional path to save raw JSON response
            
        Returns:
            StructuredOutcome object
        """
        prompt = self.construct_prompt(source_gene, target_gene, relationship, search_context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=4096,
        )
        
        if save_raw_path:
            with open(save_raw_path, 'w') as f:
                json.dump(response.model_dump(), f, indent=2)
                
        return self._parse_response(response)

    def _parse_response(self, response) -> StructuredOutcome:
        """Parse the OpenAI API response into StructuredOutcome."""
        message = response.choices[0].message
        
        # Extract usage stats
        usage_stats = None
        if hasattr(response, 'usage') and response.usage:
            # Handle standard OpenAI Usage object
            usage_stats = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        content = message.content if message.content else ""
        reasoning_content_field = getattr(message, 'reasoning_content', None)

        full_content = content.strip() if content else ""
        if not full_content and reasoning_content_field:
            full_content = reasoning_content_field.strip()

        marker = "Answer:"
        marker_pos = full_content.find(marker)

        reasoning = full_content
        answer_text = "MARKER_NOT_FOUND"
        answer_value = None

        if marker_pos >= 0:
            reasoning = full_content[:marker_pos].strip()
            # Look for answer in sub-window
            char_pos_before = marker_pos + len(marker)
            char_pos_after = min(char_pos_before + 100, len(full_content))
            answer_section = full_content[char_pos_before:char_pos_after].strip()

            answer_lines = answer_section.split('\n')
            if answer_lines:
                answer_text = marker + " " + answer_lines[0].strip()

            answer_lower = answer_text.lower()
            if 'none' in answer_lower:
                answer_value = None
            elif 'yes' in answer_lower:
                answer_value = True
            elif 'no' in answer_lower:
                answer_value = False
            else:
                answer_value = None
        else:
             lower_content = full_content.lower()
             if lower_content.startswith('yes'):
                 answer_value = True
                 answer_text = "Yes (inferred)"
             elif lower_content.startswith('no'):
                 answer_value = False
                 answer_text = "No (inferred)"
             elif lower_content.startswith('none'):
                 answer_value = None
                 answer_text = "None (inferred)"
             else:
                 answer_value = None

        return StructuredOutcome(
            reasoning=reasoning,
            answer_text=answer_text,
            answer=answer_value,
            reasoning_content=reasoning_content_field,
            usage=usage_stats
        )
