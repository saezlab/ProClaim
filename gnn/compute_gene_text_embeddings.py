import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import time


def get_bert_embedding(text, model, tokenizer):
    """
    Generates a text embedding for a given string using a pre-trained BERT model.

    Args:
        text (str): The input text to embed.
        model: The pre-trained BERT model.
        tokenizer: The pre-trained BERT tokenizer.

    Returns:
        numpy.ndarray: A 768-dimensional vector representing the text embedding.
    """
    # Tokenize the text and add special tokens ([CLS] and [SEP])
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)

    # Get the model output (no gradients needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state contains the embeddings for all tokens.
    # We will use mean pooling to get a single vector for the entire text.
    # This averages the embeddings of all tokens in the sequence.
    last_hidden_states = outputs.last_hidden_state

    # Move the tensor to the CPU and convert to a NumPy array
    embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()

    return embedding


def main():
    """
    Main function to load gene summaries, generate embeddings, and save them.
    """
    input_filename = 'data/gene_summaries.json'
    output_filename = 'data/gene_embeddings.npy'
    gene_embeddings = {}

    print("Loading pre-trained BioBERT model and tokenizer...")
    # Using BioBERT, which is pre-trained on biomedical text for better domain-specific embeddings
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
    model = BertModel.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
    print("Model loaded successfully.")

    try:
        with open(input_filename, 'r') as f:
            gene_summaries = json.load(f)

        total_genes = len(gene_summaries)
        print(f"Found {total_genes} genes in '{input_filename}'. Starting embedding generation...")

        start_time = time.time()
        for i, (gene, summary) in enumerate(gene_summaries.items()):
            # Format the text as requested
            description_text = f"Gene/Protein: {gene}\nDescription: {summary}"

            # Generate the embedding
            embedding_vector = get_bert_embedding(description_text, model, tokenizer)

            # Store the embedding as a list for JSON serialization
            gene_embeddings[gene] = embedding_vector.tolist()

            # Print progress
            print(f"Processing ({i + 1}/{total_genes}): {gene}")

        end_time = time.time()
        print(f"\nEmbedding generation completed in {end_time - start_time:.2f} seconds.")

        np.save(output_filename, gene_embeddings)
        print(f"Successfully saved embeddings to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        print("Please run 'gene_lookup.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()