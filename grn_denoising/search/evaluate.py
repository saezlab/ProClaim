import ollama
import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm
from collections import Counter
import re

# def extract_binary_with_reasoning(data):
#     binary_answers = []
    
#     for item in data:
#         if item['binary_answer'] is not None:
#             binary_answers.append(item['binary_answer'])
#         elif item['success'] and item['binary_answer'] is None:
#             reasoning = item.get('reasoning', '')
            
#             # Multiple regex patterns to catch various formats
#             patterns = [
#                 r'\*\*CONCLUSION:\*\*\s*\*\*(True|False)\*\*',  # **CONCLUSION:** **True**
#                 r'CONCLUSION:\s*\*\*(True|False)\*\*',         # CONCLUSION: **True**
#                 r'\*\*CONCLUSION:\s*(True|False)\*\*',         # **CONCLUSION: True**
#                 r'CONCLUSION:\s*(True|False)',                 # CONCLUSION: True
#                 r'CONCLUSION:\s*"(True|False)"',               # CONCLUSION: "True"
#                 r'Final Assessment[:\s]*\*\*(True|False)\*\*', # Final Assessment: **True**
#                 r'Therefore.*?(True|False)',                   # Therefore... True
#             ]
            
#             result = None
#             for pattern in patterns:
#                 match = re.search(pattern, reasoning, re.IGNORECASE)
#                 if match:
#                     result = match.group(1).lower() == 'true'
#                     break
            
#             binary_answers.append(result)
#         else:
#             binary_answers.append(None)
    
#     return binary_answers

# def extract_with_simple_parsing(data):
#     binary_answers = []
    
#     for item in data:
#         if item['binary_answer'] is not None:
#             binary_answers.append(item['binary_answer'])
#         elif item['success'] and item['binary_answer'] is None:
#             reasoning = item.get('reasoning', '').lower()
            
#             # Simple text analysis
#             if 'conclusion: true' in reasoning or 'conclusion:** true' in reasoning:
#                 binary_answers.append(True)
#             elif 'conclusion: false' in reasoning or 'conclusion:** false' in reasoning:
#                 binary_answers.append(False)
#             else:
#                 # Look for the last occurrence of true/false near conclusion words
#                 conclusion_section = reasoning.split('conclusion')[-1][:200]  # Last 200 chars after "conclusion"
#                 if 'true' in conclusion_section and 'false' not in conclusion_section:
#                     binary_answers.append(True)
#                 elif 'false' in conclusion_section and 'true' not in conclusion_section:
#                     binary_answers.append(False)
#                 else:
#                     binary_answers.append(None)
#         else:
#             binary_answers.append(None)
    
#     return binary_answers

def extract_binary_with_ollama(data):
    binary_answers = []
    
    for item in data:
        if item['binary_answer'] is not None:
            binary_answers.append(item['binary_answer'])
        elif item['success'] and item['binary_answer'] is None:
            reasoning = item.get('reasoning', '')
            
            # Try regex first (faster)
            conclusion_match = re.search(r'(?:\*\*)?CONCLUSION(?:\:|\:\*\*)\s*(?:\*\*)?(?:[""]?)(True|False)(?:[""]?)(?:\*\*)?', reasoning, re.IGNORECASE)
            if conclusion_match:
                result = conclusion_match.group(1).lower() == 'true'
                binary_answers.append(result)
            else:
                # Fall back to LLM extraction
                prompt = f"""Based on this scientific reasoning text, what is the final conclusion? Answer only "True" or "False".

Text: {reasoning[:800]}

Answer:"""
                
                try:
                    response = ollama.generate(
                        model='qwen3:8b',  # or whatever model you have installed
                        prompt=prompt
                    )
                    
                    answer_text = response['response'].strip().lower()
                    if 'true' in answer_text and 'false' not in answer_text:
                        binary_answers.append(True)
                    elif 'false' in answer_text and 'true' not in answer_text:
                        binary_answers.append(False)
                    else:
                        binary_answers.append(None)
                        
                except Exception as e:
                    print(f"LLM extraction failed: {e}")
                    binary_answers.append(None)
        else:
            binary_answers.append(None)
    
    return binary_answers

if __name__ == "__main__":
    data_path = Path('./results/with_search/')
    data_file = data_path / 'batch_search_results_3.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    binary_answers = extract_binary_with_ollama(data)
    print(Counter(binary_answers).keys())
    print(Counter(binary_answers).values())
    # data_path = Path('./results/without_search/')
    # data_file = data_path / 'batch_results_2.json'
    # with open(data_file, 'r') as f:
    #     data = json.load(f)
    # binary_answers = extract_binary_with_ollama(data)
    # print(Counter(binary_answers).keys())
    # print(Counter(binary_answers).values())