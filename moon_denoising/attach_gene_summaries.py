import requests
import json
import time


def fetch_gene_summary(gene_name):
    """
    Fetches a summary for a given gene from the NCBI Entrez database.

    Args:
        gene_name (str): The name of the gene to look up.

    Returns:
        str: The summary of the gene, or a 'not found' message.
    """
    # E-utilities base URLs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    # Parameters for esearch to find the gene ID
    esearch_params = {
        "db": "gene",
        "term": f"{gene_name}[Gene Name] AND human[Organism]",
        "retmode": "json"
    }

    try:
        # Step 1: Search for the gene to get its UID
        response = requests.get(esearch_url, params=esearch_params)
        response.raise_for_status()  # Raise an exception for bad status codes
        search_data = response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return f"Gene '{gene_name}' not found or no unique ID available."

        gene_id = id_list[0]

        # Step 2: Use the UID to fetch the gene summary
        esummary_params = {
            "db": "gene",
            "id": gene_id,
            "retmode": "json"
        }

        summary_response = requests.get(esummary_url, params=esummary_params)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        # Extract the summary from the complex JSON structure
        summary = summary_data.get("result", {}).get(gene_id, {}).get("summary", "No summary available.")

        if not summary:
            return "No summary available."

        return summary

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    except json.JSONDecodeError:
        return "Failed to decode JSON response from NCBI."


def main():
    """
    Main function to read genes, fetch summaries, and write to a JSON file.
    """
    input_filename = 'data/unique_genes.txt'
    output_filename = 'data/gene_summaries.json'
    gene_summaries = {}

    try:
        with open(input_filename, 'r') as f:
            # Read genes and strip whitespace/newlines
            genes = [line.strip() for line in f if line.strip()]

        print(f"Found {len(genes)} genes in '{input_filename}'. Starting lookup...")

        for i, gene in enumerate(genes):
            print(f"Processing ({i + 1}/{len(genes)}): {gene}")
            summary = fetch_gene_summary(gene)
            gene_summaries[gene] = summary
            # Be polite to the NCBI API by waiting a bit between requests
            time.sleep(0.1)

        with open(output_filename, 'w') as f:
            json.dump(gene_summaries, f, indent=4)

        print(f"\nSuccessfully wrote summaries to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()