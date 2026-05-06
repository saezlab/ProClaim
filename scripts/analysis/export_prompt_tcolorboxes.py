#!/usr/bin/env python3
"""Export prompt templates as LaTeX tcolorboxes.

Usage:
  uv run python scripts/analysis/export_prompt_tcolorboxes.py
    uv run python scripts/analysis/export_prompt_tcolorboxes.py --group openscholar --stdout
    uv run python scripts/analysis/export_prompt_tcolorboxes.py --mode proclaim-direct
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.baselines.shared import prompts as shared_prompts
from experiments.baselines.shared.label_utils import (
    verdict_defs_block,
    verdict_names,
    verdict_or_str,
    verdict_options_str,
)


DEFAULT_BASELINE_OUTPUT_FILE = PROJECT_ROOT / "results" / "analysis" / "prompts" / "baseline_prompt_tcolorboxes.tex"
DEFAULT_PROCLAIM_DIRECT_OUTPUT_FILE = PROJECT_ROOT / "results" / "analysis" / "prompts" / "proclaim_direct_prompt_tcolorboxes.tex"
BASELINE_GROUPS = {"shared", "react", "fire", "safe", "openscholar"}
PROCLAIM_DIRECT_GROUPS = {"orchestrator", "subagents", "search", "sufficiency"}
MODE_CHOICES = ["baseline", "proclaim-direct"]
GROUP_CHOICES = ["all", *sorted(BASELINE_GROUPS | PROCLAIM_DIRECT_GROUPS)]
TCOLORBOX_TEMPLATE = (
    "\\begin{{tcolorbox}}[breakable, colback=gray!5,colframe=blue!70,"
    "title={title}, fonttitle=\\bfseries]\n"
    "\\footnotesize\n"
    "\\begin{{Verbatim}}[breaklines,breakanywhere,showtabs=false,breaksymbol=]\n"
    "{content}\n"
    "\\end{{Verbatim}}\n"
    "\\end{{tcolorbox}}"
)


@dataclass(frozen=True)
class PromptTemplate:
    title: str
    content: str
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render local prompt templates as LaTeX tcolorboxes."
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="baseline",
        help="Prompt family to export (default: baseline).",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=GROUP_CHOICES,
        default=[],
        help="Restrict output to one or more prompt groups. Repeat to include multiple groups.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path where the LaTeX output will be written. "
            "Defaults to the mode-specific prompts file under results/analysis/prompts/."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the generated LaTeX to stdout.",
    )
    return parser.parse_args()


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latex_escape_title(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_tcolorbox(prompt: PromptTemplate) -> str:
    return TCOLORBOX_TEMPLATE.format(
        title=latex_escape_title(prompt.title),
        content=prompt.content.strip("\n"),
    )


def normalize_groups(groups: list[str]) -> set[str]:
    if not groups or "all" in groups:
        return set(BASELINE_GROUPS)
    return set(groups)


def normalize_mode_groups(mode: str, groups: list[str]) -> set[str]:
    allowed = BASELINE_GROUPS if mode == "baseline" else PROCLAIM_DIRECT_GROUPS
    if not groups or "all" in groups:
        return set(allowed)

    invalid = sorted(set(groups) - allowed)
    if invalid:
        valid_groups = ", ".join(["all", *sorted(allowed)])
        invalid_groups = ", ".join(invalid)
        raise ValueError(
            f"Mode '{mode}' does not support group(s): {invalid_groups}. Valid groups: {valid_groups}."
        )
    return set(groups)


def resolve_output_path(mode: str, output: Path | None) -> Path:
    if output is not None:
        return output
    if mode == "baseline":
        return DEFAULT_BASELINE_OUTPUT_FILE
    return DEFAULT_PROCLAIM_DIRECT_OUTPUT_FILE


def build_react_system_prompt(search_backend: str) -> str:
    verdict_defs = verdict_defs_block()
    verdict_options = verdict_or_str()
    if search_backend == "s2":
        tool_desc = (
            "- search_papers(query): Search Semantic Scholar for academic papers. "
            "Returns titles, abstracts, and metadata. Use specific queries "
            "targeting the entities and relationships in the claim."
        )
        source = "academic literature"

    return f"""You are a scientific claim verification agent using the ReAct framework.

Your task: determine whether a scientific claim is {verdict_options} based on evidence you retrieve from {source}.

Verdict definitions:
{verdict_defs}

Available tools:
{tool_desc}

Strategy:
1. Think step-by-step about what evidence you need.
2. Issue targeted search queries to find that evidence.
3. After each search result, assess whether you have enough evidence.
4. When confident, respond with your final verdict in the EXACT format:
   VERDICT: <label>
   REASONING: <one or two sentences citing key evidence>
"""

    verdict_list = ", ".join(verdict_names())
    verdict_defs = verdict_defs_block()
    verdict_options = verdict_options_str()

    generator_prompt = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list
- final_answer: your concise final answer


**Playbook:**
{playbook}

**Reflection:**
{reflection}

**Question:**
{question}

**Context:**
{context}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer here]"
}}

---
"""

    playbook = f"""## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Classify the scientific claim as {verdict_list} based on your knowledge.
[str-00002] helpful=0 harmful=0 :: Label definitions:\n{verdict_defs}
[str-00003] helpful=0 harmful=0 :: Be conservative: prefer UNCERTAIN when knowledge is insufficient.

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID
[err-00001] helpful=0 harmful=0 :: Do not hallucinate citations or evidence — rely only on your actual knowledge.

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    question_template = (
        "Scientific Claim Verification Task:\n\n"
        f"Classify the following scientific claim as one of: {verdict_list}.\n\n"
        "Claim: {claim}\n\n"
        f"Your final_answer MUST be exactly one of: {verdict_list}."
    )

    return [
        PromptTemplate("ACE Generator", generator_prompt, "ace"),
        PromptTemplate("ACE Claim Verification Playbook", playbook, "ace"),
        PromptTemplate("ACE Claim Question Template", question_template, "ace"),
        PromptTemplate(
            "ACE Final Answer Options",
            f"final_answer must be exactly one of: {verdict_options}",
            "ace",
        ),
    ]


def build_openscholar_templates() -> list[PromptTemplate]:
    instructions = load_module(
        "openscholar_instructions",
        PROJECT_ROOT / "experiments" / "OpenScholar" / "src" / "instructions.py",
    )
    task_instruction, instance_header = instructions.task_instructions["claim_verdict_question"]
    framed_claim = (
        "Is the following scientific claim supported or refuted by the literature: "
        "{claim}? Answer uncertain otherwise."
    )
    with_references = (
        f"{task_instruction}\nReferences:\n{{context}}\n{instance_header}"
        f"Is the following scientific claim supported or refuted by the literature: {{claim}}? "
        "Answer uncertain otherwise."
    )
    no_references = (
        f"{task_instruction}{instance_header}"
        f"Is the following scientific claim supported or refuted by the literature: {{claim}}? "
        "Answer uncertain otherwise."
    )

    return [
        PromptTemplate(
            "OpenScholar Claim Framing",
            framed_claim,
            "openscholar",
        ),
        # PromptTemplate(
        #     "OpenScholar Claim Verdict Question With References",
        #     with_references,
        #     "openscholar",
        # ),
        PromptTemplate(
            "OpenScholar Claim Verdict Question",
            no_references,
            "openscholar",
        ),
    ]


def build_proclaim_direct_templates(groups: set[str]) -> list[PromptTemplate]:
    from pkevolve.verification import llm_sufficiency, prompts as verification_prompts

    templates: list[PromptTemplate] = []

    if "orchestrator" in groups:
        templates.extend(
            [
                PromptTemplate(
                    "ProClaim Direct System Prompt",
                    verification_prompts.DIRECT_SYSTEM_PROMPT,
                    "orchestrator",
                ),
                PromptTemplate(
                    "ProClaim Direct Subclaim Examples",
                    verification_prompts.SUBCLAIM_EXAMPLES,
                    "orchestrator",
                ),
            ]
        )

    if "subagents" in groups:
        templates.extend(
            [
                PromptTemplate(
                    "ProClaim Direct Fact Extraction",
                    verification_prompts.EXTRACT_FACTS,
                    "subagents",
                ),
                PromptTemplate(
                    "ProClaim Direct Evidence Synthesis",
                    verification_prompts.SYNTHESIZE_SUBCLAIM,
                    "subagents",
                ),
                PromptTemplate(
                    "ProClaim Direct Conflict Detection",
                    verification_prompts.DETECT_CONFLICTS,
                    "subagents",
                ),
                PromptTemplate(
                    "ProClaim Direct Gap Identification",
                    verification_prompts.IDENTIFY_GAPS,
                    "subagents",
                ),
                PromptTemplate(
                    "ProClaim Direct Gap Query Formulation",
                    verification_prompts.FORMULATE_GAP_QUERIES,
                    "subagents",
                ),
                PromptTemplate(
                    "ProClaim Direct Failed-Paper Query Refinement",
                    verification_prompts.REFINE_SEARCH_QUERY,
                    "subagents",
                ),
            ]
        )

    if "search" in groups:
        templates.extend(
            [
                PromptTemplate(
                    "ProClaim Direct PubMed Query Generation",
                    verification_prompts.QUERY_GENERATION,
                    "search",
                ),
                PromptTemplate(
                    "ProClaim Direct Semantic Scholar Query Generation",
                    verification_prompts.QUERY_GENERATION_S2,
                    "search",
                ),
                PromptTemplate(
                    "ProClaim Direct Gap-Targeted Query Generation",
                    verification_prompts.GENERATE_GAP_QUERY,
                    "search",
                ),
            ]
        )

    if "sufficiency" in groups:
        templates.append(
            PromptTemplate(
                "ProClaim Direct LLM Sufficiency Classifier",
                llm_sufficiency.PROMPT_TEMPLATE,
                "sufficiency",
            )
        )

    return templates


def build_sufficiency_templates() -> list[PromptTemplate]:
    sufficiency = load_module(
        "sufficiency_prompts",
        PROJECT_ROOT
        / "scripts"
        / "sufficiency_classifier"
        / "slm_llm_ablation"
        / "generate_sufficiency_prompts.py",
    )
    return [
        PromptTemplate(
            "LLM Sufficiency Classifier",
            sufficiency.PROMPT_TEMPLATE,
            "sufficiency",
        )
    ]


def build_fire_templates() -> list[PromptTemplate]:
    return [
        PromptTemplate(
            "FIRE System Prompt",
            "You are a fact-checking agent responsible for verifying the accuracy of claims.",
            "fire",
        ),
        PromptTemplate(
            "FIRE Final Answer Or Next Search Prompt",
            """Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your conclusion, think through the process step-by-step. 
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. If the KNOWLEDGE allows you to confidently make a decision, output the final 
   answer as a JSON object in the following format:
   {
      \"final_answer\": \"SUPPORT\" or \"REFUTE\" or \"UNCERTAIN\"
   }
    - \"SUPPORT\" -- The retrieved evidence contains statements that directly corroborate the claim. The evidence, taken at face value, is sufficient to conclude that the claim is true or highly likely true.
    - \"REFUTE\" -- Either (a) the retrieved evidence contains statements that directly contradict the claim, or (b) given the scope of the retrieved corpus, a thorough search yields no evidence that substantiates the claim. In both cases, the evidence base does not support accepting the claim as true.
    - \"UNCERTAIN\" -- The retrieved evidence is relevant to the claim but is ambiguous, incomplete, or internally conflicting such that neither a clear supportive nor a clear refutatory conclusion can be drawn. This includes cases where evidence partially supports the claim but with meaningful caveats, or where sources of comparable credibility disagree.
5. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search 
   query that could provide additional evidence. Output the search query in JSON 
   format, as follows:
   {
     \"search_query\": \"Your Google search query here\"
   }
6. The query should aim to obtain new information not already present in the 
   KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.

KNOWLEDGE:
[KNOWLEDGE]

STATEMENT:
[STATEMENT]""",
            "fire",
        ),
    ]


def build_safe_templates() -> list[PromptTemplate]:
    return [
        PromptTemplate(
            "SAFE Next Search Prompt",
            """Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to find evidence that could support, refute, or clarify whether the given STATEMENT should remain uncertain.
3. To do this, you are allowed to issue ONE Google Search query that you think will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the KNOWLEDGE. This new information should be useful for determining the factual accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
[KNOWLEDGE]

STATEMENT:
[STATEMENT]""",
            "safe",
        ),
        PromptTemplate(
            "SAFE Final Answer Prompt",
            """Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific reasoning. As part of your reasoning, summarize the main points of the KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your final answer based on your reasoning and the STATEMENT.
6. Your final answer should be one of \"SUPPORT\", \"REFUTE\", or \"UNCERTAIN\". Wrap your final answer in square brackets.
  - \"SUPPORT\" -- The retrieved evidence contains statements that directly corroborate the claim. The evidence, taken at face value, is sufficient to conclude that the claim is true or highly likely true.
  - \"REFUTE\" -- Either (a) the retrieved evidence contains statements that directly contradict the claim, or (b) given the scope of the retrieved corpus, a thorough search yields no evidence that substantiates the claim. In both cases, the evidence base does not support accepting the claim as true.
  - \"UNCERTAIN\" -- The retrieved evidence is relevant to the claim but is ambiguous, incomplete, or internally conflicting such that neither a clear supportive nor a clear refutatory conclusion can be drawn. This includes cases where evidence partially supports the claim but with meaningful caveats, or where sources of comparable credibility disagree.

KNOWLEDGE:
[KNOWLEDGE]

STATEMENT:
[STATEMENT]""",
            "safe",
        ),
    ]


def collect_prompt_templates(mode: str, groups: set[str]) -> list[PromptTemplate]:
    if mode == "proclaim-direct":
        return build_proclaim_direct_templates(groups)

    templates: list[PromptTemplate] = []

    if "shared" in groups:
        templates.extend(
            [
                PromptTemplate(
                    "Retrieval Final Verdict System Prompt",
                    shared_prompts.VERIFICATION_SYSTEM_PROMPT,
                    "shared",
                ),
                PromptTemplate(
                    "Retrieval Final Verdict User Template",
                    shared_prompts.VERIFICATION_USER_TEMPLATE,
                    "shared",
                ),
                PromptTemplate(
                    "LLM-Only Final Verdict System Prompt",
                    shared_prompts.VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL,
                    "shared",
                ),
                PromptTemplate(
                    "LLM-Only Final Verdict User Template",
                    shared_prompts.LLM_ONLY_USER_TEMPLATE,
                    "shared",
                ),
            ]
        )

    if "react" in groups:
        templates.extend(
            [
                PromptTemplate(
                    "ReAct System Prompt (Semantic Scholar)",
                    build_react_system_prompt("s2"),
                    "react",
                ),
                PromptTemplate(
                    "ReAct User Prompt",
                    "Verify this scientific claim:\n\n{claim}",
                    "react",
                ),
            ]
        )

    if "fire" in groups:
        templates.extend(build_fire_templates())

    if "safe" in groups:
        templates.extend(build_safe_templates())

    if "openscholar" in groups:
        templates.extend(build_openscholar_templates())

    return templates


def build_document(templates: list[PromptTemplate]) -> str:
    header = [
        "% Generated by scripts/analysis/export_prompt_tcolorboxes.py",
        "% Requires: \\usepackage[most]{tcolorbox} and \\usepackage{fvextra}",
        "",
    ]
    body = [render_tcolorbox(template) for template in templates]
    return "\n\n".join(header + body) + "\n"


def main() -> None:
    args = parse_args()
    groups = normalize_mode_groups(args.mode, args.group)
    templates = collect_prompt_templates(args.mode, groups)
    document = build_document(templates)
    output_path = resolve_output_path(args.mode, args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")

    if args.stdout:
        print(document, end="")

    print(f"Wrote {len(templates)} prompt boxes to {output_path}")


if __name__ == "__main__":
    main()