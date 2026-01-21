import json
import numpy as np
from pathlib import Path

def get_token_stats(base_dir):
    paths = list(Path(base_dir).rglob("*.json"))
    total_tokens_list = []
    
    print(f"Scanning {len(paths)} files in {base_dir}...")
    
    for p in paths:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
                
            res = data.get('result', {})
            
            # Try to get real total tokens
            real_usage = res.get('token_usage_real')
            if real_usage and isinstance(real_usage, dict):
                total = real_usage.get('total_tokens')
                if total:
                    total_tokens_list.append(total)
                    continue
            
            # Fallback to token_usage (which might be prompt only based on previous read) if real is missing?
            # Or token_usage_estimate
            # Let's stick to files that have real usage for accurate stats
            
        except Exception as e:
            # print(f"Error reading {p}: {e}")
            pass

    if not total_tokens_list:
        print("No valid token usage data found.")
        return

    tokens = np.array(total_tokens_list)
    
    mean_val = np.mean(tokens)
    median_val = np.median(tokens)
    min_val = np.min(tokens)
    max_val = np.max(tokens)
    
    print(f"Count: {len(tokens)}")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Interval (Min - Max): {min_val} - {max_val}")

if __name__ == "__main__":
    get_token_stats("results/qa_comprehensive/gpt-oss-120b/full_text/rater")
