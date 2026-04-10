#!/usr/bin/env python3
"""Quick smoke test for a locally running vLLM server."""

import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Smoke test a local vLLM server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None, help="Model name (default: first available)")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    # List available models
    models = client.models.list().data
    if not models:
        print("ERROR: No models available on the server.")
        return

    model_id = args.model or models[0].id
    print(f"Server: http://localhost:{args.port}/v1")
    print(f"Model:  {model_id}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=256,
    )

    print(response.choices[0].message.content)
    print("-" * 40)
    usage = response.usage
    print(f"Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")

if __name__ == "__main__":
    main()
