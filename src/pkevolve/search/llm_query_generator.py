"""
LLM-driven query generation for PubMed search.

This module provides functionality to generate comprehensive PubMed queries
using LLMs, replacing the entity-based extraction approach with a more flexible
LLM-driven strategy that works for diverse claim types (PPI, diagnosis, drug resistance, etc.).
"""

from typing import Callable
import logging

from pkevolve.verification.prompts import (
    QUERY_GENERATION as QUERY_GENERATION_PROMPT,
    QUERY_GENERATION_S2 as QUERY_GENERATION_S2_PROMPT,
    GENERATE_GAP_QUERY,
)

logger = logging.getLogger(__name__)


def _validate_query(query: str) -> bool:
    """
    Check for common PubMed syntax errors.

    Args:
        query: The PubMed query string to validate

    Returns:
        True if query appears valid, False otherwise
    """
    # Check for unbalanced parentheses
    if query.count("(") != query.count(")"):
        logger.warning(f"Unbalanced parentheses in query: {query}")
        return False

    # Check for invalid operators (common LLM mistakes)
    if " AN " in query or " O " in query:  # Should be AND/OR
        logger.warning(f"Invalid operators in query: {query}")
        return False

    # Warn if very short or very long
    word_count = len(query.split())
    if word_count < 2:
        logger.warning(f"Query too short ({word_count} words): {query}")
        return False
    if word_count > 30:
        logger.warning(f"Query may be too long ({word_count} words): {query}")
        # Don't fail for being too long, just warn

    return True


def _sanitize_query(query: str) -> str:
    """
    Fix common LLM query mistakes and formatting issues.

    Args:
        query: The raw query string from LLM

    Returns:
        Sanitized query string
    """
    import re

    # Remove <think>...</think> tags and their content (common in reasoning models)
    query = re.sub(r'<think>.*?</think>', '', query, flags=re.DOTALL)

    # Remove other common XML-style reasoning tags
    query = re.sub(r'<reasoning>.*?</reasoning>', '', query, flags=re.DOTALL)
    query = re.sub(r'<explanation>.*?</explanation>', '', query, flags=re.DOTALL)

    # Remove markdown code block markers
    query = query.strip("`").strip()

    # Remove "Query:" prefix if present
    if query.lower().startswith("query:"):
        query = query[6:].strip()

    # Fix lowercase Boolean operators
    query = query.replace(" and ", " AND ")
    query = query.replace(" or ", " OR ")
    query = query.replace(" not ", " NOT ")

    # Fix unmatched brackets
    if query.count("[") != query.count("]"):
        # Remove all brackets if unmatched
        query = query.replace("[", "").replace("]", "")
        logger.warning(f"Removed unmatched brackets from query")

    # Remove leading/trailing spaces and newlines
    query = query.strip()

    return query


def generate_search_query(
    claim: str,
    llm: Callable,
    max_attempts: int = 2,
    subclaims: list[str] | None = None,
    extraction_context: list[str] | None = None,
) -> str:
    """
    LLM-driven query formulation for initial PubMed search.

    Args:
        claim: The scientific claim to verify
        llm: LLM callable that takes a prompt string and returns a response string
        max_attempts: Maximum number of attempts to generate a valid query
        subclaims: Optional subclaims; when provided, the prompt includes them so
            the sub-agent can pick up aliases introduced during decomposition.
        extraction_context: Optional disambiguation/synonym notes to incorporate
            into the query generation prompt.

    Returns:
        PubMed query string

    Raises:
        ValueError: If unable to generate a valid query after max_attempts
    """
    subclaims_section = ""
    if subclaims:
        joined = "\n".join(f"- {s}" for s in subclaims)
        subclaims_section = f"\n\nSubclaims (use these to pick up aliases and alternative names):\n{joined}"

    context_section = ""
    if extraction_context:
        notes = "\n".join(f"- {n}" for n in extraction_context)
        context_section = f"\n\nSupplementary context (use to refine query terms):\n{notes}"

    for attempt in range(max_attempts):
        try:
            # Generate query using LLM
            prompt = QUERY_GENERATION_PROMPT.format(claim=claim) + subclaims_section + context_section
            response = llm(prompt)

            # Extract query from response (handle various formats)
            query = response.strip()

            # Sanitize common issues
            query = _sanitize_query(query)

            # Validate
            if _validate_query(query):
                logger.info(f"[LLM Query] Generated: {query}")
                return query
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts}: Invalid query generated")
                if attempt < max_attempts - 1:
                    # Try to fix simple issues
                    continue

        except Exception as e:
            logger.error(f"Error generating query on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                continue
            raise

    # If all attempts failed, create a simple fallback query
    logger.warning(f"Failed to generate valid query after {max_attempts} attempts. Using fallback.")
    # Simple fallback: extract words that look like entities (capitalized terms)
    words = claim.split()
    entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
    if len(entities) >= 2:
        fallback = " AND ".join(entities[:4])  # Take first 4 entities
        logger.info(f"[Fallback Query] {fallback}")
        return fallback
    else:
        # Ultimate fallback: use the whole claim (first 10 words)
        fallback = " ".join(claim.split()[:10])
        logger.info(f"[Fallback Query] {fallback}")
        return fallback


def generate_search_query_s2(
    claim: str,
    llm: Callable,
    subclaims: list[str] | None = None,
    max_attempts: int = 2,
    extraction_context: list[str] | None = None,
) -> str:
    """LLM-driven query formulation for Semantic Scholar search.

    Args:
        claim: The scientific claim to verify
        llm: LLM callable that takes a prompt string and returns a response string
        subclaims: Optional subclaims for alias-enriched query generation.
        max_attempts: Maximum number of attempts to generate a valid query
        extraction_context: Optional disambiguation/synonym notes to incorporate
            into the query generation prompt.

    Returns:
        Semantic Scholar query string (plain keywords, no PubMed field tags)
    """
    subclaims_section = ""
    if subclaims:
        joined = "\n".join(f"- {s}" for s in subclaims)
        subclaims_section = f"\n\nSubclaims (use these to pick up aliases and alternative names):\n{joined}"

    context_section = ""
    if extraction_context:
        notes = "\n".join(f"- {n}" for n in extraction_context)
        context_section = f"\n\nSupplementary context (use to refine query terms):\n{notes}"

    for attempt in range(max_attempts):
        try:
            prompt = QUERY_GENERATION_S2_PROMPT.format(
                claim=claim,
                subclaims_section=subclaims_section,
            ) + context_section
            response = llm(prompt)
            query = _sanitize_query(response.strip())
            if _validate_query(query):
                logger.info(f"[S2 Query] Generated: {query}")
                return query
            else:
                logger.warning(f"S2 attempt {attempt + 1}/{max_attempts}: invalid query")
        except Exception as e:
            logger.error(f"Error generating S2 query on attempt {attempt + 1}: {e}")
            if attempt >= max_attempts - 1:
                raise

    # Fallback: capitalized entity words from claim
    words = claim.split()
    entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
    fallback = " ".join(entities[:5]) if len(entities) >= 2 else " ".join(claim.split()[:8])
    logger.warning(f"S2 fallback query: {fallback}")
    return fallback


def generate_gap_query(
    claim: str,
    gap_description: str,
    llm: Callable,
) -> str:
    """
    Generate a targeted PubMed query for filling an evidence gap.

    This is used when initial search is insufficient and specific gaps
    have been identified that need targeted searches.

    Args:
        claim: The original scientific claim
        gap_description: Description of what evidence is missing
        llm: LLM callable

    Returns:
        Targeted PubMed query string
    """
    prompt = GENERATE_GAP_QUERY.format(
        claim=claim,
        gap_description=gap_description,
    )

    try:
        response = llm(prompt)
        query = _sanitize_query(response.strip())

        if _validate_query(query):
            logger.info(f"[Gap Query] Generated: {query}")
            return query
        else:
            # Fallback: use gap description key terms
            logger.warning("Gap query validation failed, using gap description")
            return gap_description

    except Exception as e:
        logger.error(f"Error generating gap query: {e}")
        return gap_description
