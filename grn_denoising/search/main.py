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
import random

def load_config():
    """Load configuration from file with fallback to defaults"""
    default_config = {
        'model': 'gpt-oss:20b',
        'searxng_instances': [
            'http://localhost:8889'
        ],
        'max_results': 8,
        'timeout': 50,
        'max_retries': 3,
        'history_file': 'search_history.json',
        'enable_colors': True,
        'streaming_delay': 0.02,
        'request_delay': 2.0,
        'ncbi_delay': 3.0,
        'searxng_delay': 3.0,
        'exclude_ncbi_domains': False,  # Set to True to exclude NCBI/PMC results
        'ncbi_domains': [
            'ncbi.nlm.nih.gov',
            'pubmed.ncbi.nlm.nih.gov',
            'pmc.ncbi.nlm.nih.gov'
        ]
    }
    
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
    except Exception as e:
        print(f"Warning: Could not load config.json, using defaults: {e}")
    
    return default_config

CONFIG = load_config()

class WebSearchAssistant:
    def __init__(self, enable_history: bool = True, verbose: bool = False, exclude_ncbi: Optional[bool] = None):
        """
        Initialize WebSearchAssistant
        
        Args:
            enable_history: Whether to load/save search history
            verbose: Whether to print status messages
            exclude_ncbi: Override config setting to exclude NCBI domains (None uses config value)
        """
        self.verbose = verbose
        self.enable_history = enable_history
        
        # Allow runtime override of NCBI exclusion
        if exclude_ncbi is not None:
            self.exclude_ncbi = exclude_ncbi
        else:
            self.exclude_ncbi = CONFIG.get('exclude_ncbi_domains', False)
        
        self.ncbi_domains = CONFIG.get('ncbi_domains', [
            'ncbi.nlm.nih.gov',
            'pubmed.ncbi.nlm.nih.gov',
            'pmc.ncbi.nlm.nih.gov'
        ])
        
        # Load search history if enabled
        if self.enable_history:
            self.search_history = self.load_history()
        else:
            self.search_history = []
        
        # Setup HTTP session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        # Track last request time for rate limiting
        self.last_request_time = {}
        
        if self.verbose:
            print(f"WebSearchAssistant initialized - Model: {CONFIG['model']}")
            if self.exclude_ncbi:
                print(f"⚠️  NCBI/PMC domains will be excluded from results")

    def _is_ncbi_domain(self, url: str) -> bool:
        """
        Check if URL belongs to NCBI/PMC domains
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from NCBI/PMC domain
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        return any(ncbi_domain in domain for ncbi_domain in self.ncbi_domains)

    def _filter_ncbi_results(self, results: List[Dict]) -> List[Dict]:
        """
        Filter out NCBI/PMC results if configured to do so
        
        Args:
            results: List of search results
            
        Returns:
            Filtered list of results
        """
        if not self.exclude_ncbi:
            return results
        
        filtered = [r for r in results if not self._is_ncbi_domain(r.get('url', ''))]
        
        if self.verbose and len(filtered) < len(results):
            excluded_count = len(results) - len(filtered)
            print(f"🚫 Excluded {excluded_count} NCBI/PMC result(s)")
        
        return filtered

    def _rate_limit(self, domain: str, is_ncbi: bool = False):
        """
        Ensure sufficient delay between requests to the same domain
        
        Args:
            domain: Domain name to rate limit
            is_ncbi: Whether this is an NCBI domain (uses longer delay)
        """
        delay = CONFIG.get('ncbi_delay', 3.0) if is_ncbi else CONFIG.get('request_delay', 2.0)
        delay = random.uniform(delay - 1.0, delay + 1.0)  # Add jitter
        
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < delay:
                wait_time = delay - elapsed
                if self.verbose:
                    print(f"Rate limiting: waiting {wait_time:.1f}s for {domain}")
                time.sleep(wait_time)
        
        self.last_request_time[domain] = time.time()

    def model_response(self, model: str, message: str, max_retries: int = None) -> Optional[str]:
        """
        Get response from Ollama model with error handling
        
        Args:
            model: Ollama model name
            message: Prompt/message to send to model
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Model response text or None if failed
        """
        if max_retries is None:
            max_retries = CONFIG.get('max_retries', 3)
            
        for attempt in range(max_retries):
            try:
                response = ollama.chat(model=model, messages=[
                {
                    'role': 'system',
                    'content': 'You are a molecular biologist expert in biological interaction.',
                },
                {
                    'role': 'user',
                    'content': message,
                }])
                return response['message']['content']
                
            except Exception as e:
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Exponential backoff for retries
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    if self.verbose:
                        print(f"Failed to get model response after {max_retries} attempts")
                    return None

    def load_history(self) -> List[Dict]:
        """Load search history from file"""
        try:
            if os.path.exists(CONFIG['history_file']):
                with open(CONFIG['history_file'], 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def browse_web(self, query: str, after_year: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Search the web using multiple SearxNG instances with fallback
        """
        if self.verbose:
            print(f"Searching web for: {query}")
            if after_year:
                print(f"Filtering results after year: {after_year}")
        
        # Add year filter to query if specified
        search_query = query
        if after_year:
            search_query = f"{query} after:{after_year}"
        
        for instance in CONFIG['searxng_instances']:
            try:
                # SearXNG rate limiting
                parsed = urlparse(instance)
                searxng_delay = CONFIG.get('searxng_delay', 3.0)
                
                # SearXNG delay
                if parsed.netloc in self.last_request_time:
                    elapsed = time.time() - self.last_request_time[parsed.netloc]
                    if elapsed < searxng_delay:
                        wait_time = searxng_delay - elapsed
                        if self.verbose:
                            print(f"Rate limiting SearXNG: waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                
                self.last_request_time[parsed.netloc] = time.time()
                
                search_url = f"{instance}/search?q={search_query}&format=json&categories=general"
                
                response = self.session.get(search_url, timeout=CONFIG['timeout'])
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                
                # Filter NCBI results if configured
                results = self._filter_ncbi_results(results)
                
                if results:
                    limited_results = results[:CONFIG['max_results']]
                    if self.verbose:
                        print(f"Found {len(limited_results)} results from {instance}")
                    return limited_results
                        
            except Exception as e:
                if self.verbose:
                    print(f"Search instance {instance} failed: {str(e)}")
                continue
                    
        if self.verbose:
            print("All search instances failed")
        return None
    
    def retrieve_page_information(self, url: str) -> Optional[str]:
        """
        Retrieve and clean webpage content using Jina Reader API
        
        Args:
            url: URL to retrieve content from
            
        Returns:
            Cleaned webpage content or None if failed
        """
        try:
            if self.verbose:
                print(f"Extracting content from: {url}")
            
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Parse URL to check domain and apply appropriate rate limiting
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            is_ncbi = self._is_ncbi_domain(url)

            # Apply rate limiting before making request
            self._rate_limit(domain, is_ncbi=is_ncbi)
                
            base_url = "https://r.jina.ai/"
            response = self.session.get(
                base_url + url, 
                timeout=CONFIG['timeout'],
                headers={'Accept': 'text/plain'}
            )
            response.raise_for_status()
            
            content = response.text.strip()
            
            # Limit content size to prevent token overflow
            if len(content) > 10000:
                content = content[:10000] + "\n... (content truncated)"
                
            if self.verbose:
                print(f"Retrieved {len(content)} characters of content")
                
            return content
            
        except requests.exceptions.Timeout:
            if self.verbose:
                print("Timeout: Page took too long to respond")
            return None
        except requests.exceptions.ConnectionError:
            if self.verbose:
                print("Connection error: Cannot reach the website")
            return None
        except requests.exceptions.HTTPError as e:
            if self.verbose:
                print(f"HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve page content: {str(e)}")
            return None
        
    def select_best_result(self, question: str, results: List[Dict]) -> Optional[Tuple[str, str]]:
        """
        Use AI to select the most relevant search result
        
        Args:
            question: Original user question (also used as search query)
            results: List of search results from browse_web()
            
        Returns:
            Tuple of (title, url) for best result, or None if failed
        """
        results_text = "\n".join([
            f"{i+1}. {result.get('title', 'No title')} - {result.get('url', 'No URL')}\n   {result.get('content', 'No description')[:200]}..."
            for i, result in enumerate(results)
        ])
        
        prompt = f"""You are an expert at evaluating search results. Based on the original question, select the MOST RELEVANT result.

Question: {question}

Search Results:
{results_text}

Respond with ONLY the title and URL in this exact format:
Title: [exact title from results]
URL: [exact URL from results]"""

        if self.verbose:
            print("AI selecting best result...")
            
        response = self.model_response(CONFIG['model'], prompt)
        
        if not response:
            if self.verbose:
                print("Failed to get AI result selection")
            return None
            
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            title = next((line.split(':', 1)[1].strip() for line in lines if line.startswith('Title:')), None)
            url = next((line.split(':', 1)[1].strip() for line in lines if line.startswith('URL:')), None)
            
            if title and url:
                if self.verbose:
                    print(f"Selected: {title}")
                return title, url
                
        except Exception as e:
            if self.verbose:
                print(f"Error parsing result selection: {e}")
            
        # Fallback to first result if AI selection fails
        if results:
            fallback_title = results[0].get('title', 'Unknown')
            fallback_url = results[0].get('url', '')
            if self.verbose:
                print(f"Using fallback result: {fallback_title}")
            return fallback_title, fallback_url
            
        return None
    
    def generate_reasoning_and_binary_answer(self, question: str, title: str, content: str) -> Tuple[Optional[str], Optional[bool]]:
        """
        Generate comprehensive reasoning and extract binary answer
        
        Args:
            question: Original user question
            title: Title of the selected source
            content: Retrieved webpage content
            
        Returns:
            Tuple of (reasoning, binary_answer) where:
            - reasoning: Full reasoning text or None if failed
            - binary_answer: True/False or None if couldn't extract
        """
        prompt = f"""You are a molecular biology expert analyzing biological interactions based on scientific literature.

Question: {question}
Source: {title}

Retrieved Content:
{content}

Instructions:
1. Provide a comprehensive analysis of the biological interaction described in the question
2. Use information from the retrieved content to support your reasoning
3. Consider the molecular mechanisms, pathways, and experimental evidence
4. Discuss any conflicting evidence or uncertainty
5. Be thorough in your scientific reasoning

After your detailed analysis, provide your final assessment in this exact format:
CONCLUSION: [True/False]

Where True means there is strong scientific evidence supporting the interaction, and False means there is insufficient evidence or evidence against the interaction."""

        if self.verbose:
            print("Generating reasoning and binary answer...")
            
        response = self.model_response(CONFIG['model'], prompt)
        
        if not response:
            return None, None
        
        # Extract binary answer from the response
        binary_answer = self._extract_binary_conclusion(response)
        
        return response, binary_answer
    
    def _extract_binary_conclusion(self, text: str) -> Optional[bool]:
        """
        Extract binary conclusion from model response
        
        Args:
            text: Full model response text
            
        Returns:
            True/False or None if couldn't extract
        """
        if not text:
            return None
            
        # Look for CONCLUSION: True/False with optional markdown formatting
        conclusion_pattern = r'CONCLUSION:\s*\*?\*?(True|False)\*?\*?'
        match = re.search(conclusion_pattern, text, re.IGNORECASE)
        
        if match:
            conclusion_text = match.group(1).lower()
            return conclusion_text == 'true'
            
        # Fallback: look for final True/False at end of text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1].lower()
            if last_line == 'true':
                return True
            elif last_line == 'false':
                return False
        
        # Additional fallback: look for "answer: true/false" pattern
        answer_pattern = r'answer:\s*(true|false)'
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == 'true'
        
        return None

    def save_search_to_history(self, question: str, result: Dict) -> bool:
        """
        Save search to history file
        
        Args:
            question: Original user question
            result: Dictionary containing search result info (title, url, reasoning, binary_answer)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enable_history:
            return True  # Return True if history is disabled (not an error)
            
        try:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'result': result
            }
            
            self.search_history.append(entry)
            
            # Keep only last 50 searches to prevent file from growing too large
            if len(self.search_history) > 50:
                self.search_history = self.search_history[-50:]
                
            with open(CONFIG['history_file'], 'w') as f:
                json.dump(self.search_history, f, indent=2)
                
            if self.verbose:
                print(f"Saved search to history: {CONFIG['history_file']}")
                
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Could not save history: {e}")
            return False
    
    def search_and_answer(self, question: str) -> Dict:
        """
        Complete search workflow - from question to reasoning and binary answer
        
        Args:
            question: User's question to search for
            
        Returns:
            Dictionary containing:
            - success: bool - Whether the search was successful
            - question: str - Original question
            - reasoning: str - Full reasoning text (if successful)
            - binary_answer: bool - True/False conclusion (if successful)
            - source_title: str - Title of selected source (if successful)
            - source_url: str - URL of selected source (if successful)
            - error: str - Error message (if failed)
            - timestamp: str - When the search was performed
        """
        result = {
            'success': False,
            'question': question,
            'reasoning': None,
            'binary_answer': None,
            'source_title': None,
            'source_url': None,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if self.verbose:
                print(f"Starting search for: {question}")
            
            # Step 1: Search the web using question directly as query
            search_results = self.browse_web(question, after_year=2018)
            if not search_results:
                result['error'] = "No search results found"
                return result
            # Step 2: Select best result using AI
            selected_result = self.select_best_result(question, search_results)
            if not selected_result:
                result['error'] = "Could not select a relevant result"
                return result
                
            title, url = selected_result
            result['source_title'] = title
            result['source_url'] = url
            
            # Step 3: Retrieve page content
            content = self.retrieve_page_information(url)
            if not content:
                result['error'] = "Could not retrieve page content"
                return result
            
            # Step 4: Generate reasoning and binary answer
            reasoning, binary_answer = self.generate_reasoning_and_binary_answer(question, title, content)
            if not reasoning:
                result['error'] = "Could not generate reasoning"
                return result
                
            result['reasoning'] = reasoning
            result['binary_answer'] = binary_answer
            result['success'] = True
            
            # Step 5: Save to history (optional)
            if self.enable_history:
                history_data = {
                    'title': title,
                    'url': url,
                    'reasoning': reasoning,
                    'binary_answer': binary_answer
                }
                self.save_search_to_history(question, history_data)
            
            if self.verbose:
                print("Search completed successfully")
                if binary_answer is not None:
                    print(f"Binary conclusion: {binary_answer}")
                else:
                    print("Warning: Could not extract binary conclusion")
                
            return result
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            if self.verbose:
                print(f"Search failed: {str(e)}")
            return result



if __name__ == "__main__":
    # Initialize assistant with NCBI exclusion
    # Option 1: Use config file setting
    assistant = WebSearchAssistant(verbose=False, enable_history=False)
    
    # Option 2: Override at runtime to exclude NCBI
    # assistant = WebSearchAssistant(verbose=False, enable_history=False, exclude_ncbi=True)
    
    # Option 3: Override at runtime to include NCBI
    # assistant = WebSearchAssistant(verbose=False, enable_history=False, exclude_ncbi=False)

    # Load signor negative edges csv file
    data_path = Path('../all_removed_edges_with_sources.csv')
    data = pd.read_csv(data_path)
    sources = data['ENTITYA'].tolist()
    targets = data['ENTITYB'].tolist()
    interactions = data['EFFECT'].tolist()

    repeat = 20
    for r in tqdm(range(14, repeat)):
        results = []
        # for i in range(1):
        for i in range(len(sources)):
            source = sources[i]
            target = targets[i]
            interaction = interactions[i]

            if interaction == 'down-regulates':
                question = f"Does {source} down-regulate {target}?"
            elif interaction == 'down-regulates activity':
                question = f"Does {source} inhibit the activity of {target}?"
            elif interaction == 'form complex':
                question = f"Does {source} form a complex with {target}?"
            elif interaction == 'up-regulates':
                question = f"Does {source} up-regulate {target}?"
            elif interaction == 'up-regulates activity':
                question = f"Does {source} activate {target}?"
            elif interaction == 'up-regulates quantity':
                question = f"Does {source} increase {target} expression?"
            elif interaction == 'up-regulates quantity by expression':
                question = f"Does {source} increase {target} expression?"
            else:  # handles 'unknown' and any other unexpected interactions
                question = f"Does {source} interact with {target}?"

            result = assistant.search_and_answer(question)
            results.append(result)
            
            # if result['success']:
            #     print(f"✅ Success: {result['question']}")
            #     print(f"📄 Source: {result['source_title']}")
            #     print(f"🔬 Binary Answer: {result['binary_answer']}")
            #     print(f"💡 Reasoning Preview: {result['reasoning'][:150]}...")
            # else:
            #     print(f"❌ Failed: {result['error']}")
            
            # print("-" * 80)
            
            # Save all results to file after each iteration
            with open(f'./results/with_search/batch_search_results_{r}.json', 'w') as f:
                json.dump(results, f, indent=2)