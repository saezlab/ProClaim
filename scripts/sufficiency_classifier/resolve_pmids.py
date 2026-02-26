"""
Resolve PMIDs for SciFact evidence documents using NCBI Entrez API.

Multi-strategy search:
  1. Exact title [Title] search
  2. Keyword search (first N significant words)
  3. Abstract-based search (for bad titles like copyright notices)

Caches doc_id -> PMID mapping to avoid repeated lookups.

Usage:
    uv run scripts/sufficiency_classifier/resolve_pmids.py
    uv run scripts/sufficiency_classifier/resolve_pmids.py --limit 10   # test with 10 docs
    uv run scripts/sufficiency_classifier/resolve_pmids.py --retry-failed  # retry only failed entries
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = PROJECT_ROOT / "data" / "doc_id_to_pmid_cache.json"
CLAIMS_PATH = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
CORPUS_PATH = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl")

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def load_env() -> dict:
    """Load .env file for PubMed credentials."""
    env_path = PROJECT_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip("\"'")
    return env_vars


def _entrez_search(term: str, email: str, api_key: str = None, retmax: int = 3) -> list[str]:
    """Execute an Entrez esearch and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{ENTREZ_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SciFact-PMID-Resolver/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode()
        root = ET.fromstring(data)
        return [e.text for e in root.findall(".//Id")]
    except Exception as e:
        logging.warning(f"Entrez search error: {e}")
        return []


def _is_bad_title(title: str) -> bool:
    """Detect titles that are not actual paper titles (copyright notices, URLs, etc.)."""
    bad_patterns = [
        r"^From\s+\w+\.\w+",          # "From bloodjournal.hematologylibrary.org ..."
        r"personal use only",
        r"copyright",
        r"^http",
        r"doi\.org",
    ]
    for p in bad_patterns:
        if re.search(p, title, re.IGNORECASE):
            return True
    return False


def _extract_keyword_query(title: str, n_words: int = 8) -> str:
    """Extract the first N significant words from a title for keyword search."""
    # Remove special characters that break Entrez queries
    cleaned = re.sub(r'[^\w\s-]', ' ', title)
    words = cleaned.split()
    # Skip very short/common words
    significant = [w for w in words if len(w) > 2]
    return " ".join(significant[:n_words])


def _extract_abstract_query(abstract: list[str], n_words: int = 12) -> str:
    """Extract key phrases from abstract for search."""
    if not abstract:
        return ""
    # Use first sentence (usually most descriptive)
    first_sentence = abstract[0] if abstract else ""
    cleaned = re.sub(r'[^\w\s-]', ' ', first_sentence)
    words = cleaned.split()
    significant = [w for w in words if len(w) > 3]
    return " ".join(significant[:n_words])


def resolve_pmid(doc: dict, email: str, api_key: str = None, delay: float = 0.11) -> tuple[str | None, str]:
    """
    Multi-strategy PMID resolution.

    Returns (pmid, strategy_used) where strategy is one of:
    'title_exact', 'title_keyword', 'abstract', or 'not_found'.
    """
    title = doc.get("title", "")
    abstract = doc.get("abstract", [])

    # Strategy 1: Exact title search (unless title is garbage)
    if title and not _is_bad_title(title):
        ids = _entrez_search(f'{title}[Title]', email, api_key, retmax=1)
        if ids:
            return ids[0], "title_exact"
        time.sleep(delay)

        # Strategy 2: Keyword search from title
        keyword_q = _extract_keyword_query(title)
        if keyword_q and len(keyword_q.split()) >= 4:
            ids = _entrez_search(keyword_q, email, api_key, retmax=3)
            if ids:
                return ids[0], "title_keyword"
            time.sleep(delay)

    # Strategy 3: Abstract-based search
    if abstract:
        abstract_q = _extract_abstract_query(abstract)
        if abstract_q and len(abstract_q.split()) >= 5:
            ids = _entrez_search(abstract_q, email, api_key, retmax=3)
            if ids:
                return ids[0], "abstract"
            time.sleep(delay)

    return None, "not_found"


def main():
    parser = argparse.ArgumentParser(description="Resolve PMIDs for SciFact evidence documents")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs to resolve (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-resolve all, ignoring existing cache")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry entries with empty PMID")
    parser.add_argument("--all-corpus", action="store_true",
                        help="Resolve PMIDs for ALL corpus docs (not just evidence). "
                             "Needed for noise paper metadata in training data.")
    args = parser.parse_args()

    # Load credentials
    env_vars = load_env()
    email = env_vars.get("PUBMED_EMAIL")
    api_key = env_vars.get("PUBMED_API_KEY")

    if not email:
        logging.error("PUBMED_EMAIL not found in .env")
        sys.exit(1)

    logging.info(f"Using email: {email}")
    logging.info(f"API key: {'yes' if api_key else 'no (rate limited to 3 req/s)'}")

    # Load existing cache
    cache = {}
    if CACHE_FILE.exists() and not args.force:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        logging.info(f"Loaded existing cache with {len(cache)} entries")

    # Load corpus
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            did = str(doc["doc_id"])
            corpus[did] = doc

    logging.info(f"Loaded corpus: {len(corpus)} documents")

    # Determine target doc_ids
    if args.all_corpus:
        target_doc_ids = set(corpus.keys())
        logging.info(f"Mode: ALL corpus docs ({len(target_doc_ids)})")
    else:
        # Only evidence doc_ids from claims
        with open(CLAIMS_PATH) as f:
            claims = [json.loads(l) for l in f]
        target_doc_ids = set()
        for c in claims:
            for did in c.get("evidence", {}).keys():
                target_doc_ids.add(str(did))
        target_doc_ids = target_doc_ids.intersection(corpus.keys())
        logging.info(f"Mode: evidence docs only ({len(target_doc_ids)} in corpus)")

    # Filter to docs that need resolution
    to_resolve = []
    for did in target_doc_ids:
        if args.retry_failed:
            if did in cache and not cache[did]:
                to_resolve.append(did)
        elif args.force or did not in cache or not cache[did]:
            to_resolve.append(did)

    logging.info(f"Need to resolve: {len(to_resolve)} docs ({sum(1 for v in cache.values() if v)} already resolved)")

    if args.limit:
        to_resolve = to_resolve[:args.limit]
        logging.info(f"Limited to {len(to_resolve)} docs for testing")

    delay = 0.11 if api_key else 0.35
    strategy_counts = {"title_exact": 0, "title_keyword": 0, "abstract": 0, "not_found": 0}
    resolved = 0

    for i, did in enumerate(to_resolve):
        doc = corpus[did]
        title = doc.get("title", "N/A")

        pmid, strategy = resolve_pmid(doc, email, api_key, delay)
        strategy_counts[strategy] += 1

        if pmid:
            cache[did] = pmid
            resolved += 1
            logging.info(f"[{i+1}/{len(to_resolve)}] Doc {did} -> PMID {pmid} ({strategy})")
        else:
            cache[did] = ""
            logging.warning(f"[{i+1}/{len(to_resolve)}] Doc {did}: not found ('{title[:50]}...')")

        # Save periodically
        if (i + 1) % 50 == 0:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
            logging.info(f"Saved checkpoint ({len(cache)} entries)")

        time.sleep(delay)

    # Final save
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    total_with_pmid = sum(1 for v in cache.values() if v)
    logging.info(f"\nDone! Newly resolved: {resolved}/{len(to_resolve)}")
    logging.info(f"Strategy breakdown: {strategy_counts}")
    logging.info(f"Total cache: {len(cache)} entries, {total_with_pmid} with PMIDs")
    logging.info(f"Saved to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
