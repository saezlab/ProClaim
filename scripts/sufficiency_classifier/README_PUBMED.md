# PubMed MCP Server Integration

This directory uses `mcp-simple-pubmed` (via `uvx`) instead of the default Anthropic PubMed plugin.

## Why?
The original Anthropic PubMed plugin (`plugins/pubmed`) frequently fails with "Rate exceeded" errors and reliability issues (issue #35). We switched to another MCP server to ensure stable access to PubMed and PMC.

## MCP Server Used
*   **Name**: `mcp-simple-pubmed`
*   **Repository**: [andybrandt/mcp-simple-pubmed](https://github.com/andybrandt/mcp-simple-pubmed)
*   **Type**: Python-based MCP server

## Configuration
The server runs automatically when `extract_features_scifact.py` is executed.

### Environment Variables
*   `PUBMED_EMAIL` (Required, set in `.env`)
*   `PUBMED_API_KEY` (Optional, set in `.env`)

## Usage
```bash
uv run python scripts/sufficiency_classifier/extract_features_scifact.py --limit 3
```
