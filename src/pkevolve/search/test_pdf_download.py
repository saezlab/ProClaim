#!/usr/bin/env python3
"""Simple test script to download PDFs from paper searches."""

import asyncio
import os
from datetime import datetime
from paper_search_mcp.server import search_pubmed, download_pubmed, download_arxiv
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher
from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher

async def test_pubmed_download():
    """Test PubMed search and PDF download."""
    print("\n=== Testing PubMed Search and PDF Download ===")

    query = "BCL2L1 activate BAD"
    print(f"Query: {query}\n")

    # Search PubMed
    searcher = PubMedSearcher()
    papers = searcher.search(query, max_results=5)

    print(f"Found {len(papers)} papers\n")

    # Create output directory
    pdf_dir = "pdfs/pubmed"
    os.makedirs(pdf_dir, exist_ok=True)

    # Try to download PDFs
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] Title: {paper.title[:80]}...")
        print(f"    PMID: {paper.paper_id}")
        print(f"    URL: {paper.url}")

        try:
            result = await download_pubmed(paper.paper_id, pdf_dir)
            print(f"    Download result: {result}")
        except Exception as e:
            print(f"    Download failed: {e}")

    print(f"\n✓ PubMed test complete. PDFs saved to: {os.path.abspath(pdf_dir)}")

async def test_arxiv_download():
    """Test arXiv search and PDF download."""
    print("\n\n=== Testing arXiv Search and PDF Download ===")

    query = "gene regulatory network inference"
    print(f"Query: {query}\n")

    # Search arXiv
    searcher = ArxivSearcher()
    papers = searcher.search(query, max_results=3)

    print(f"Found {len(papers)} papers\n")

    # Create output directory
    pdf_dir = "pdfs/arxiv"
    os.makedirs(pdf_dir, exist_ok=True)

    # Download PDFs
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] Title: {paper.title[:80]}...")
        print(f"    arXiv ID: {paper.paper_id}")
        print(f"    URL: {paper.url}")

        try:
            result = await download_arxiv(paper.paper_id, pdf_dir)
            print(f"    ✓ Downloaded to: {result}")
        except Exception as e:
            print(f"    ✗ Download failed: {e}")

    print(f"\n✓ arXiv test complete. PDFs saved to: {os.path.abspath(pdf_dir)}")

async def main():
    """Run all tests."""
    print("=" * 70)
    print("Paper PDF Download Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Test PubMed
    await test_pubmed_download()

    # Test arXiv
    await test_arxiv_download()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
