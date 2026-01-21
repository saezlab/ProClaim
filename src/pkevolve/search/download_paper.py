#!/usr/bin/env python3
"""
CLI script to download papers using paper-search-mcp.
Usage:
    python download_paper.py --id <paper_id> --source <arxiv|pubmed> [--output <dir>]
"""

import argparse
import asyncio
import os
import sys
from paper_search_mcp.server import download_arxiv, download_pubmed, download_biorxiv, download_medrxiv

async def download_paper(paper_id: str, source: str, output_dir: str):
    """Download a paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {source} paper: {paper_id} to {output_dir}...")
    
    try:
        if source == 'arxiv':
            result = await download_arxiv(paper_id, output_dir)
            print(f"✓ Success: {result}")
        elif source == 'pubmed':
            result = await download_pubmed(paper_id, output_dir)
            print(f"Result: {result}")
        elif source == 'biorxiv':
            result = await download_biorxiv(paper_id, output_dir)
            print(f"✓ Success: {result}")
        elif source == 'medrxiv':
            result = await download_medrxiv(paper_id, output_dir)
            print(f"✓ Success: {result}")
        else:
            print(f"Error: Unknown source '{source}'")
            return
            
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download papers using paper-search-mcp")
    parser.add_argument("--id", required=True, help="Paper ID (e.g., 2106.12345 for arXiv, 12345678 for PubMed, or DOI for bioRxiv/medRxiv)")
    parser.add_argument("--source", required=True, choices=['arxiv', 'pubmed', 'biorxiv', 'medrxiv'], help="Source of the paper")
    parser.add_argument("--output", default="pdfs", help="Output directory (default: pdfs)")
    
    args = parser.parse_args()
    
    asyncio.run(download_paper(args.id, args.source, args.output))

if __name__ == "__main__":
    main()
