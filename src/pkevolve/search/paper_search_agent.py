
import os
import json
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import glob
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from pkevolve.search.paper_utils import (
    get_paper_details, 
    download_pdf_from_doi, extract_text_from_pdf,
    get_pmc_fulltext, get_fulltext_from_jina
)
from pkevolve.search.custom_pubmed import RelevancePubMedSearcher

# ============================================================================
# LLM Setup
# ============================================================================

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="gpt-oss-120b",
    temperature=0
)

# ============================================================================
# Functions
# ============================================================================

def validate_text_quality(text: Optional[str], source: str, verbose: bool = False) -> bool:
    """
    Validate if the text is likely a full paper using heuristics and LLM.
    """
    if not text:
        if verbose: print(f"[WARN] No text captured from {source}.")
        return False
        
    # 1. Heuristic Checks
    if len(text) < 3000:
        if verbose: print(f"[WARN] Text from {source} is too short ({len(text)} chars). Likely strict abstract or error.")
        return False
        
    error_keywords = [
        "Access to this page has been denied", 
        "403 Forbidden", 
        "JavaScript is disabled",
        "Enable JavaScript and cookies",
        "Just a moment...",
        "Verify you are human"
    ]
    if any(k in text for k in error_keywords):
        if verbose: print(f"[WARN] Text from {source} contains error keywords.")
        return False

    # 2. LLM Judge
    if verbose: print(f"[INFO] Validating text quality from {source} with LLM...")
    prompt = f"""
You are a quality assurance agent for scientific literature.
Determine if the following text represents a **FULL SCIENTIFIC PAPER** or just an abstract/fragment/junk.

Criteria for FULL PAPER:
- Contains multiple sections (Introduction, Methods, Results, Discussion).
- significant length and detail.
- Not just a list of references or a landing page.

Text Sample (First 2000 chars):
{text[:2000]}

...

Text Sample (Middle 2000 chars):
{text[len(text)//2 : len(text)//2 + 2000]}

Respond with only "YES" if it is a full paper, or "NO" if it is not.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        judgement = response.content.strip().upper()
        if "YES" in judgement:
            if verbose: print(f"[INFO] LLM validated text from {source} as FULL PAPER.")
            return True
        else:
            if verbose: print(f"[WARN] LLM rejected text from {source}. Judgement: {judgement}")
            return False
    except Exception as e:
        if verbose: print(f"[ERROR] LLM validation failed: {e}. Assuming text is valid based on length.")
        return True # Fallback if LLM fails but heuristics passed

def retrieve_best_full_text(paper_details: Dict[str, Any], auto: bool = False, verbose: bool = False) -> Optional[str]:
    """
    Try multiple methods to get high-quality full text.
    Order: DOI PDF -> PMC -> Jina -> User Manual Input
    """
    pmid = paper_details.get('pmid')
    doi = paper_details.get('doi')
    pmcid = paper_details.get('pmcid')
    existing_text = paper_details.get('full_text')
    
    # 0. Check existing text (e.g. from PMC via get_paper_details)
    if existing_text and validate_text_quality(existing_text, "Initial Fetch", verbose=verbose):
        return existing_text
        
    if verbose: print(f"\n[INFO] Starting retrieval for PMID: {pmid} / DOI: {doi}")
    
    # ensure directories exist
    # Use absolute path relative to project root
    # src/pkevolve/search/paper_search_agent.py -> search -> pkevolve -> src -> root
    project_root = Path(__file__).resolve().parents[3]
    pdf_dir = project_root / "data/papers/qa_search/pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Try DOI PDF Download
    if doi:
        if verbose: print(f"[STEP 1] Attempting PDF download for DOI: {doi}")
        pdf_path = download_pdf_from_doi(doi, save_dir=str(pdf_dir))
        if pdf_path:
            text = extract_text_from_pdf(pdf_path)
            if validate_text_quality(text, "DOI PDF", verbose=verbose):
                return text
        else:
            if verbose: print("[INFO] DOI download failed or no open access PDF found.")
            
    # 2. Try PMC (Explicit Retry)
    if pmcid:
        if verbose: print(f"[STEP 2] Attempting PMC full text for: {pmcid}")
        text = get_pmc_fulltext(pmcid)
        if validate_text_quality(text, "PMC API", verbose=verbose):
            return text
            
    # 3. Try Jina AI (Web Scraping)
    target_url = paper_details.get('doi_url') or paper_details.get('url')
    if target_url:
        if verbose: print(f"[STEP 3] Attempting Jina AI for URL: {target_url}")
        text = get_fulltext_from_jina(target_url)
        if validate_text_quality(text, "Jina AI", verbose=verbose):
            return text


    # 4. Interactive Fallback
    if auto:
        if verbose: print(f"[INFO] Auto mode: No valid full text found (or validation failed) for {pmid}. Setting full_text to None.")
        return None

    print(f"\\n" + "="*60)
    print(f"[ALERT] Could not automatically retrieve full text for:")
    print(f"Title: {paper_details.get('title')}")
    print(f"Link: {paper_details.get('doi_url') or paper_details.get('url')}")
    print("="*60)
    
    # Always verbose for interactive prompt
    verbose = True
    
    while True:
        choice = input("Select action: [P]ath to PDF file | [S]kip paper | [Q]uit agent: ").strip().lower()
        
        if choice == 'q':
            print("[INFO] Quitting agent...")
            exit(0)
            
        elif choice == 's':
            print("[INFO] Skipping paper.")
            return ""
            
        elif choice == 'p':
            raw_input = input("Enter absolute path to PDF file: ")
            pdf_path_input = raw_input.strip().strip("'").strip('"')
            print(f"[DEBUG] User provided path: {pdf_path_input}")
            
            if os.path.exists(pdf_path_input):
                print(f"[DEBUG] File exists. Extracting text...")
                try:
                    text = extract_text_from_pdf(pdf_path_input)
                    print(f"[DEBUG] Extraction complete. Length: {len(text)} chars. Validating...")
                    
                    if validate_text_quality(text, "User PDF"):
                        print(f"[INFO] User PDF accepted. Returning text and proceeding to next paper...")
                        return text
                    else:
                        print("[WARN] Extracted text from provided PDF failed validation. Try again or skip.")
                except Exception as e:
                    print(f"[ERROR] Failed to read PDF: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[ERROR] File does not exist at: {pdf_path_input}")
        else:
            print(f"Invalid choice: '{choice}'")

def convert_question_to_query(question: str, verbose: bool = False) -> str:
    """
    Convert a natural language question into a MINIMAL PubMed search query.
    Attempt 1: Strict intersection of the entities mentioned in the question.
    """
    prompt = f"""
You are a helpful biomedical research assistant.
Convert the following question into a PubMed search query.

Goal: Find papers that mention BOTH entities relevant to the question.
Strategy: Create a STRICT INITIAL query using ONLY the specific entites mentioned in the question.

CRITICAL GUIDELINES:
1. Identify the two key entities (Entity A and Entity B) from the question.
2. Use ONLY the names explicitly found in the question (or their primary standard symbol if the name is colloquial, e.g. "Torc2" -> "CRTC2").
3. Do NOT add synonyms yet.
4. Do NOT add interaction terms (e.g., "activate", "inhibit", "regulate").
5. Use AND to combine the two entities.

Question: "{question}"

Output only the query string inside <query> tags.

Example:
Question: "Does CRTC2 upregulate AKT1?"
<query>
(CRTC2) AND (AKT1)
</query>
"""
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    if verbose: print(f"[DEBUG] Raw LLM response (Initial): {content}")
    
    import re
    match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
    if match:
        query = match.group(1).strip()
    else:
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        query = clean_content.strip('"').strip("'")
        
    if verbose: print(f"[INFO] Converted question to query (Initial): {query}")
    return query

def generate_broad_query(previous_query: str, attempt: int, verbose: bool = False) -> str:
    """
    Generate a broader query iteratively by adding ONE new search term per entity.
    Each attempt adds one more synonym/alternative name to each entity.
    """

    prompt = f"""
You are a biomedical expert helping to broaden a PubMed search query step-by-step.
The previous query did not yield enough results. Your task is to add ONE new search term for EACH entity.

Previous Query: "{previous_query}"
Current Broadening Attempt: {attempt}

TASK:
- Analyze the previous query to identify the two entities (Entity A and Entity B)
- For EACH entity, add exactly ONE new alternative name/synonym to its OR group
- The new term should be the next most common/useful synonym not yet included

STRATEGY for selecting the new term:
- Start with the most specific and commonly used synonyms (e.g., PKB for AKT1, AC1 for ADCY1)
- Then add full protein names (e.g., "protein kinase B" for AKT1)
- Then add alternative gene symbols or abbreviations
- Finally, if all specific terms are exhausted, add broader family terms (but only as last resort)

CRITICAL RULES:
1. Add EXACTLY ONE new term per entity (not zero, not multiple)
2. Do NOT repeat terms already in the previous query
3. Maintain the structure: (Entity A terms) AND (Entity B terms)
4. Do NOT add interaction terms (like "activate", "inhibit", "regulate")
5. Output ONLY the new query inside <query> tags

EXAMPLES:

Example 1:
Previous Query: (GNAO1) AND (ADCY1)
Analysis: No synonyms yet for either entity. Add the most common synonym to each.
New Query:
<query>
(GNAO1 OR "G protein subunit alpha o1") AND (ADCY1 OR AC1)
</query>

Example 2:
Previous Query: (GNAO1 OR "G protein subunit alpha o1") AND (ADCY1 OR AC1)
Analysis: Each entity has 1 synonym. Add one more to each.
New Query:
<query>
(GNAO1 OR "G protein subunit alpha o1" OR "Gαo") AND (ADCY1 OR AC1 OR "adenylate cyclase type 1")
</query>

Example 3:
Previous Query: (CRTC2) AND (AKT1)
Analysis: No synonyms yet. Add most common synonym to each.
New Query:
<query>
(CRTC2 OR TORC2) AND (AKT1 OR PKB)
</query>

Example 4:
Previous Query: (CRTC2 OR TORC2) AND (AKT1 OR PKB)
Analysis: Each has 1 synonym. Add one more to each.
New Query:
<query>
(CRTC2 OR TORC2 OR "CREB regulated transcription coactivator 2") AND (AKT1 OR PKB OR "protein kinase B")
</query>

Now generate the broadened query for the given previous query.
Remember: Add EXACTLY ONE new term to each entity's OR group.
"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content.strip()

    import re
    match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
    if match:
        query = match.group(1).strip()
    else:
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        query = clean_content.strip('"').strip("'")

    if verbose: print(f"[INFO] Generated BROADER query (Attempt {attempt}): {query}")
    return query

def save_single_result(result: Dict[str, Any], index: int, query: str, output_base: str, original_question: str, save_dir: Optional[Path] = None, verbose: bool = False):
    """
    Save a single result immediately.
    """
    if save_dir is None:
        # Resolve path relative to project root (now using this file's location in src/pkevolve/search)
        # We want to go up to project root: src -> pkevolve -> search -> [file]
        # So parent = search, parent.parent = pkevolve, parent.parent.parent = src, parent.parent.parent.parent = project_root
        # Actually it's simpler: scripts usually invoke this, but default relative to CWD might be better?
        # Let's keep the logic relative to the file location to be consistent, but "project root" logic changes.
        
        # Current file: .../src/pkevolve/search/paper_search_agent.py
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3] # search -> pkevolve -> src -> root
        save_dir = project_root / "data/papers/qa_search"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{output_base}_{index}.json"
    save_path = save_dir / filename
    
    # Add query/question to result
    result['pubmed_query'] = query
    result['question'] = original_question

    def json_serial(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, default=json_serial, indent=2)
        if verbose: print(f"[INFO] Saved result {index} to {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save result to {save_path}: {e}")

def run_paper_search_agent(question: str, output_base: str, target_full_text_count: int = 5, max_attempts: int = 10, auto: bool = False, save_dir: Optional[Path] = None, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Agentic search loop:
    1. Define initial query.
    2. Check existing results.
    3. If count < target, search.
    4. Search.
    5. Retrieve full text.
    6. If count < target, broaden query and retry.
    """
    # 0. Setup directories and check existing
    if save_dir is None:
        # Should match logic in save_single_result generally, but here we define it authoritative
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        save_dir = project_root / "data/papers/qa_search"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    collected_papers = {} # PMID -> PaperDetails

    # Check for existing files
    if verbose: print(f"[INFO] Checking for existing files in {save_dir} with base '{output_base}'...")
    
    # Simple pattern match
    search_pattern = f"{output_base}_*.json"
    existing_files = list(save_dir.glob(search_pattern))
    max_index = -1
    
    for fpath in existing_files:
        try:
             # parsing index from filename suffix _(\d+).json
             match = re.search(r'_(\d+)\.json$', fpath.name)
             if match:
                 idx = int(match.group(1))
                 if idx > max_index:
                     max_index = idx
             
             with open(fpath, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 # Add to collected papers
                 paper_id = data.get('pmid') or data.get('paper_id')
                 if paper_id: 
                     collected_papers[str(paper_id)] = data
        except Exception as e:
            if verbose: print(f"[WARN] Failed to load existing file {fpath}: {e}")

    valid_full_text_count = sum(1 for p in collected_papers.values() if p.get('full_text'))
    if verbose: print(f"[STATUS] Found {len(collected_papers)} processed papers. Valid Full-Text: {valid_full_text_count}/{target_full_text_count}")

    if valid_full_text_count >= target_full_text_count:
        if verbose: print(f"[SUCCESS] Target full-text count reached with existing files. Skipping search.")
        return list(collected_papers.values())

    # Proceed to search
    next_file_index = max_index + 1
    current_query = convert_question_to_query(question, verbose=verbose)
    
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        if verbose:
            print(f"\\n{'='*80}")
            print(f"[AGENT] Attempt {attempt}/{max_attempts} | Target Full-Text: {target_full_text_count}")
            print(f"[AGENT] Current Query: {current_query}")
            print(f"{'='*80}")
        
        # 1. Search PubMed
        searcher = RelevancePubMedSearcher()
        try:
            # Increase max_results per search to have a better chance of finding full text
            results_batch = searcher.search(current_query, max_results=30) 
        except Exception as e:
            if verbose: print(f"[ERROR] PubMed search failed: {e}")
            results_batch = []
            
        if verbose: print(f"[INFO] Found {len(results_batch)} papers in this batch.")
        
        # 2. Process Batch
        new_papers_count = 0
        for i, paper in enumerate(results_batch):
            # Check if we already processed this paper
            if str(paper.paper_id) in collected_papers:
                continue
            
            if verbose: print(f"\\n[INFO] Processing paper {i+1}/{len(results_batch)} (PMID: {paper.paper_id})")
            
            # Fetch details
            details = get_paper_details(paper.paper_id)
            
            # Retrieve full text
            full_text = retrieve_best_full_text(details, auto=auto, verbose=verbose)
            details['full_text'] = full_text
            
            # Store
            collected_papers[str(paper.paper_id)] = details
            new_papers_count += 1
            
            # Incremental Save
            # Use next_file_index to ensure we don't overwrite existing
            save_single_result(details, next_file_index, current_query, output_base, question, save_dir=save_dir, verbose=verbose)
            next_file_index += 1
            
            # Check if we have enough full-text papers
            valid_full_text_count = sum(1 for p in collected_papers.values() if p.get('full_text'))
            if verbose: print(f"[STATUS] Total Valid Full-Text Papers: {valid_full_text_count}/{target_full_text_count}")
            
            if valid_full_text_count >= target_full_text_count:
                if verbose: print(f"\\n[SUCCESS] Reached target of {target_full_text_count} full-text papers!")
                return list(collected_papers.values())
        
        if new_papers_count == 0:
            if verbose: print("[WARN] No NEW papers found in this attempt.")
            
        # 3. Check Status and Plan Next Step
        valid_full_text_count = sum(1 for p in collected_papers.values() if p.get('full_text'))
        if valid_full_text_count >= target_full_text_count:
            return list(collected_papers.values())
            
        if verbose: print(f"[AGENT] Only have {valid_full_text_count} full-text papers. Broadening search...")
        
        # 4. Generate Broader Query
        current_query = generate_broad_query(current_query, attempt, verbose=verbose)
        
    if verbose: print(f"[AGENT] Max attempts reached. Returning {len(collected_papers)} papers.")
    return list(collected_papers.values())
