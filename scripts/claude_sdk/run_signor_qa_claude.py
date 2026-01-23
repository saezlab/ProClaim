"""
Batch test Signor edges using Claude API.

Usage:
    # Test single edge per label
    uv run python run_signor_qa_claude.py --test-single
    
    # Process first 10 edges per label
    uv run python run_signor_qa_claude.py --max-edges 10

    # Select model (default: claude-sonnet-4-20250514)
    uv run python run_signor_qa_claude.py --model claude-sonnet-4-5-20250929
    
    # Process only true_positive edges
    uv run python run_signor_qa_claude.py --label true_positive
    
    # Run specific edge(s) - useful for re-running failed cases
    uv run python run_signor_qa_claude.py --edge CRTC2_AKT1_up-regulates --label true_negative
    uv run python run_signor_qa_claude.py --edge GNAS_ADCY1_up_regulates_activity --label true_positive
    
    # Run full batch (all edges)
    uv run python run_signor_qa_claude.py

This script:
1. Uses Claude model via Anthropic API
2. Reads Signor data from CSV files (true_positive_edges.csv, true_negative_edges.csv)
3. Constructs questions for each edge using construct_signor_question
4. Runs questions through Claude with access to PubMed plugin and edge_curation skill
5. Saves results to label-based folders (true_positive, true_negative)
"""

import asyncio
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock, ResultMessage
import sys

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from pkevolve.utils.signor_utils import load_signor_data, construct_signor_question


async def run_single_edge(options, source: str, target: str, effect: str, run_id: str, label: str):
    """
    Run a single edge through the Claude SDK with full tool capabilities.
    
    Args:
        options: ClaudeAgentOptions with PubMed plugin and skill support
        source: Source protein
        target: Target protein
        effect: Interaction effect
        run_id: Unique identifier for this run
        label: Label (true_positive or true_negative)
    
    Returns:
        Dict with question, response, and metadata
    """
    # Construct question using signor_utils
    question = construct_signor_question(source, target, effect)
    # Add "Validate" prefix to trigger the ppi-curation skill
    question_with_instruction = f"Validate this protein interaction: {question} Answer a single TRUE or FALSE at the end of your response."
    
    responses = []

    tool_calls = []  # Tool use requests from LLM
    tool_results = []  # Tool execution results
    all_messages = []
    usage_data = {}  # Token usage information
    start_time = datetime.now()
    
    try:
        async for message in query(prompt=question_with_instruction, options=options):
            message_data = {
                "type": type(message).__name__,
                "content": []
            }
            
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    block_info = {
                        "type": type(block).__name__
                    }
                    
                    if hasattr(block, 'text'):
                        block_info["text"] = block.text
                        responses.append(block.text)
                    
                    if hasattr(block, 'thinking'):
                        block_info["thinking"] = block.thinking
                    
                    if isinstance(block, ToolUseBlock):
                        block_info["tool_name"] = block.name
                        block_info["tool_id"] = block.id
                        block_info["tool_input"] = block.input  # Capture tool input parameters
                        tool_calls.append({
                            "name": block.name,
                            "id": block.id,
                            "input": block.input  # Save complete input
                        })
                    
                    message_data["content"].append(block_info)
            
            elif isinstance(message, UserMessage):
                # Capture tool results returned from tool execution
                if hasattr(message, 'content') and message.content:
                    for block in message.content:
                        block_info = {
                            "type": type(block).__name__
                        }
                        
                        if isinstance(block, ToolResultBlock):
                            # Safely convert content to string and limit size
                            content_str = str(block.content) if block.content else ""
                            # Limit to first 10000 chars to prevent memory/serialization issues
                            if len(content_str) > 10000:
                                content_str = content_str[:10000] + "... [truncated]"
                            
                            block_info["tool_use_id"] = block.tool_use_id
                            block_info["content"] = content_str
                            block_info["is_error"] = getattr(block, 'is_error', False)
                            
                            tool_results.append({
                                "tool_use_id": block.tool_use_id,
                                "content": content_str,
                                "is_error": getattr(block, 'is_error', False)
                            })
                        
                        message_data["content"].append(block_info)
            
            elif isinstance(message, ResultMessage):
                # ResultMessage contains the usage information
                if hasattr(message, 'usage') and message.usage:
                    usage_data = {
                        "input_tokens": message.usage.get('input_tokens', 0) if isinstance(message.usage, dict) else getattr(message.usage, 'input_tokens', 0),
                        "output_tokens": message.usage.get('output_tokens', 0) if isinstance(message.usage, dict) else getattr(message.usage, 'output_tokens', 0),
                        "cache_creation_input_tokens": message.usage.get('cache_creation_input_tokens', 0) if isinstance(message.usage, dict) else getattr(message.usage, 'cache_creation_input_tokens', 0),
                        "cache_read_input_tokens": message.usage.get('cache_read_input_tokens', 0) if isinstance(message.usage, dict) else getattr(message.usage, 'cache_read_input_tokens', 0),
                    }
            
            all_messages.append(message_data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        full_response = "\n".join(responses)
        
        # Extract final answer (TRUE/FALSE)
        # Look for standalone TRUE/FALSE in the last few lines to avoid picking up
        # mentions in the reasoning text
        final_answer = None
        lines = full_response.strip().split('\n')
        
        # Check last 10 lines in reverse order
        for line in reversed(lines[-10:]):
            line_clean = line.strip()
            # Skip empty lines
            if not line_clean:
                continue
            
            # Look for standalone TRUE or FALSE (case-insensitive, word boundary)
            # This will match: TRUE, FALSE, **TRUE**, **FALSE**, etc.
            matches = re.findall(r'\b(TRUE|FALSE)\b', line_clean.upper())
            if matches:
                # Take the last match in this line as the final answer
                final_answer = matches[-1]
                break
        
        # Fallback to simple logic if nothing found
        if not final_answer:
            response_upper = full_response.upper()
            if "TRUE" in response_upper and "FALSE" in response_upper:
                true_pos = response_upper.rfind("TRUE")
                false_pos = response_upper.rfind("FALSE")
                final_answer = "TRUE" if true_pos > false_pos else "FALSE"
            elif "TRUE" in response_upper:
                final_answer = "TRUE"
            elif "FALSE" in response_upper:
                final_answer = "FALSE"
        
        # Display token usage
        usage_str = ""
        if usage_data:
            total_tokens = usage_data.get('input_tokens', 0) + usage_data.get('output_tokens', 0)
            usage_str = f" | Tokens: {total_tokens} (in: {usage_data.get('input_tokens', 0)}, out: {usage_data.get('output_tokens', 0)})"
            if usage_data.get('cache_read_input_tokens', 0) > 0:
                usage_str += f" | Cache read: {usage_data.get('cache_read_input_tokens', 0)}"
        
        print(f"  ✅ Completed in {duration:.2f}s | Answer: {final_answer}{usage_str}")
        

        
        return {
            "run_id": run_id,
            "label": label,
            "source": source,
            "target": target,
            "effect": effect,
            "question": question,
            "question_with_instruction": question_with_instruction,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,

            "response": full_response,    # Final text response
            "final_answer": final_answer,
            "tool_calls": tool_calls,     # Detailed tool use requests with inputs
            "tool_results": tool_results,  # Tool execution results with outputs
            "raw_messages": all_messages,  # Complete message history
            "usage": usage_data if usage_data else None,  # Token usage statistics
            "success": True
        }
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"  ❌ Error: {e}")
        return {
            "run_id": run_id,
            "label": label,
            "source": source,
            "target": target,
            "effect": effect,
            "question": question,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "error": str(e),
            "success": False
        }


async def main():
    parser = argparse.ArgumentParser(description="Batch test Signor edges with Claude model")
    parser.add_argument("--test-single", action="store_true", help="Test only first edge from each label")
    parser.add_argument("--max-edges", type=int, help="Maximum edges to test per label")
    parser.add_argument("--label", choices=["true_positive", "true_negative"], help="Test only specific label")
    parser.add_argument("--edge", type=str, help="Run specific edge in format: SOURCE_TARGET_EFFECT (e.g., CRTC2_AKT1_up-regulates)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Claude model to use")
    args = parser.parse_args()
    
    # Validate arguments
    if args.edge and not args.label:
        parser.error("--edge requires --label to be specified")
    
    if args.edge and (args.test_single or args.max_edges):
        parser.error("--edge cannot be used with --test-single or --max-edges")
    
    # Load .env
    env_vars = {}
    possible_paths = [
        ".env",
        "../../.env",
        Path(__file__).resolve().parent.parent.parent / ".env"
    ]
    
    env_path = None
    for p in possible_paths:
        if isinstance(p, str):
            p = Path(p)
        if p.exists():
            env_path = p
            break
    
    if not env_path:
        print("Error: .env file not found")
        return
    
    print(f"Loading .env from: {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                env_vars[key.strip()] = value
    
    # Claude model configuration
    model_name = args.model
    print(f"Using Claude model: {model_name}")
    
    auth_token = env_vars.get("CLAUDE_API_KEY")
        
    if not auth_token:
        print("Error: CLAUDE_API_KEY not found in .env")
        return

    import os
    # Merge with current environment to preserve PATH and other vars
    full_env = os.environ.copy()

    # Environment configuration for Claude
    env_config = {
        "API_TIMEOUT_MS": env_vars.get("API_TIMEOUT_MS", "3000000"),
        "ANTHROPIC_API_KEY": auth_token
    }
    
    print("Using API endpoint: Default Anthropic API")
    
    # Get project root directory (not test directory) for skill loading
    project_root = Path(__file__).resolve().parent.parent.parent
    current_dir = Path(__file__).resolve().parent
    
    # Verify .claude/skills exists
    claude_skills_dir = project_root / ".claude" / "skills"
    if not claude_skills_dir.exists():
        print(f"⚠️  WARNING: {claude_skills_dir} does not exist!")
        print("Skills should be in .claude/skills/ at project root")
    else:
        print(f"✓ Skills directory found: {claude_skills_dir}")
    
    full_env.update(env_config)
    
    # Verify access to Claude model
    print(f"Verifying access to model: {model_name}...")
    try:
        import urllib.request
        
        headers = {
            "x-api-key": auth_token,
            "anthropic-version": "2023-06-01"
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models",
            headers=headers
        )
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                available_models = [m['id'] for m in data.get('data', [])]
                
                if model_name not in available_models:
                    print(f"\n❌ Error: Model '{model_name}' not found in your available Claude models.")
                    print(f"   Available models (first 5): {', '.join(available_models[:5])}...")
                    sonnet_models = [m for m in available_models if 'sonnet' in m]
                    if sonnet_models:
                        print(f"   Did you mean one of these? {', '.join(sonnet_models)}")
                    return
                print(f"✓ Model '{model_name}' appears valid.")
        except urllib.error.HTTPError as e:
            print(f"⚠️  Warning: Could not list models (HTTP {e.code}). Proceeding anyway, but run might fail.")
        except Exception as e:
            print(f"⚠️  Warning: Could not list models ({e}). Proceeding anyway.")
            
    except ImportError:
        pass

    # Configure options with PubMed plugin and skill support (let Claude use all tools)
    options_kwargs = {
        "cwd": str(current_dir),  # Use test directory where .claude/settings.local.json exists
        "env": full_env,
        "setting_sources": ["project", "local"],  # Load skills (project) and PubMed plugin (local)
        "allowed_tools": [
            "Skill",  # For edge_curation skill
            # PubMed MCP plugin tools
            "mcp__plugin_pubmed_PubMed__search_articles",
            "mcp__plugin_pubmed_PubMed__get_article_metadata",
            "mcp__plugin_pubmed_PubMed__find_related_articles",
            "mcp__plugin_pubmed_PubMed__get_full_text_article",
            "mcp__plugin_pubmed_PubMed__convert_article_ids",
            "mcp__plugin_pubmed_PubMed__lookup_article_by_citation",
            "mcp__plugin_pubmed_PubMed__get_copyright_status",
        ],
        # Only disallow tools that could cause data leakage or user interaction
        "disallowed_tools": [
            "Write",  # Prevent writing files that could leak info to subsequent runs
            "AskUserQuestion",  # No user interaction during batch testing
            "WebSearch",  # Force use of PubMed MCP plugin instead of web scraping
        ],
    }
    
    if model_name:
        options_kwargs["model"] = model_name
    
    options = ClaudeAgentOptions(**options_kwargs)
    
    # Load Signor data
    data_signor_base = PROJECT_ROOT / "data/signor"
    
    labels_to_process = []
    if args.label:
        labels_to_process = [args.label]
    else:
        labels_to_process = ["true_positive", "true_negative"]
    
    print(f"\n{'='*70}")
    print(f"BATCH TEST - Signor Edges (Full Capabilities)")
    print(f"{'='*70}")
    print(f"Labels to process: {', '.join(labels_to_process)}")
    print(f"Test mode: {'SINGLE TEST' if args.test_single else 'FULL BATCH'}")
    if args.max_edges:
        print(f"Max edges per label: {args.max_edges}")
    print(f"Features: PubMed plugin, edge_curation skill, web search")
    print(f"{'='*70}\n")
    
    batch_start = datetime.now()
    all_results = []
    
    # Create output directory structure
    results_base = PROJECT_ROOT / "results" / "claude_sdk" / "signor_skill_pubmed_search" / model_name
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Process each label
    for label in labels_to_process:
        csv_file = f"{label}_edges.csv"
        csv_path = data_signor_base / csv_file
        
        if not csv_path.exists():
            print(f"⚠️  Skipping {label}: CSV file not found at {csv_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {label}")
        print(f"{'='*70}")
        
        # Load data
        df = load_signor_data(str(csv_path), columns=['ENTITYA', 'ENTITYB', 'EFFECT'])
        
        if df is None or df.empty:
            print(f"⚠️  No data found in {csv_path}")
            continue
        
        # Filter for specific edge if --edge is provided
        if args.edge:
            # Parse edge format: SOURCE_TARGET_EFFECT
            parts = args.edge.split('_')
            if len(parts) < 3:
                print(f"⚠️  Invalid edge format: {args.edge}")
                print(f"    Expected format: SOURCE_TARGET_EFFECT (e.g., CRTC2_AKT1_up-regulates)")
                continue
            
            edge_source = parts[0]
            edge_target = parts[1]
            # Convert underscores to spaces for matching (handles both 'up_regulates_activity' and 'up-regulates_activity')
            edge_effect = ' '.join(parts[2:]).replace('_', ' ')
            
            # Try exact match first
            df_filtered = df[
                (df['ENTITYA'].astype(str) == edge_source) & 
                (df['ENTITYB'].astype(str) == edge_target) & 
                (df['EFFECT'].astype(str) == edge_effect)
            ]
            
            # If not found, try normalizing both sides (replace hyphens and underscores with spaces)
            if df_filtered.empty:
                def normalize_effect(s):
                    return s.replace('-', ' ').replace('_', ' ').lower().strip()
                
                edge_effect_norm = normalize_effect(edge_effect)
                df_filtered = df[
                    (df['ENTITYA'].astype(str) == edge_source) & 
                    (df['ENTITYB'].astype(str) == edge_target) & 
                    (df['EFFECT'].astype(str).apply(normalize_effect) == edge_effect_norm)
                ]
            
            if df_filtered.empty:
                print(f"⚠️  Edge not found: {edge_source} -> {edge_target} ({edge_effect})")
                print(f"    Available edges in {label}:")
                for _, row in df.head(5).iterrows():
                    print(f"      {row['ENTITYA']}_{row['ENTITYB']}_{str(row['EFFECT']).replace(' ', '_').replace('-', '_')}")
                continue
            
            df = df_filtered
            print(f"Found specific edge: {edge_source} -> {edge_target} ({df.iloc[0]['EFFECT']})")
        
        # Determine how many edges to process
        num_edges = len(df)
        if args.test_single and not args.edge:
            num_edges = 1
        elif args.max_edges and not args.edge:
            num_edges = min(num_edges, args.max_edges)
        
        print(f"Processing {num_edges} edge{'s' if num_edges > 1 else ''} from {label}")
        
        # Process each edge
        for idx, (_, row) in enumerate(df.head(num_edges).iterrows()):
            source = str(row['ENTITYA'])
            target = str(row['ENTITYB'])
            effect = str(row['EFFECT'])
            
            run_id = f"{label}_{idx+1}_{source}_{target}"
            
            print(f"\n[{idx+1}/{num_edges}] {source} -> {target} ({effect})")
            
            result = await run_single_edge(options, source, target, effect, run_id, label)
            result["model"] = model_name
            all_results.append(result)
            
            # Save individual result immediately
            label_dir = results_base / label
            label_dir.mkdir(exist_ok=True)
            
            # Create filename from edge info
            source_clean = result["source"].replace(" ", "_")
            target_clean = result["target"].replace(" ", "_")
            effect_clean = result["effect"].replace(" ", "_").replace("-", "_")
            filename = f"{source_clean}_{target_clean}_{effect_clean}.json"
            
            output_path = label_dir / filename
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Wait between runs (except for the last one)
            if idx < num_edges - 1:
                wait_time = 2
                await asyncio.sleep(wait_time)
        
        print(f"\n✅ Completed {label}: {num_edges} edge{'s' if num_edges > 1 else ''} processed\n")
    
    batch_end = datetime.now()
    batch_duration = (batch_end - batch_start).total_seconds()
    
    # Calculate statistics per label
    label_stats = {}
    for label in labels_to_process:
        label_results = [r for r in all_results if r["label"] == label]
        successful = [r for r in label_results if r["success"]]
        failed = [r for r in label_results if not r["success"]]
        
        if successful:
            durations = [r["duration_seconds"] for r in successful]
            label_stats[label] = {
                "total": len(label_results),
                "successful": len(successful),
                "failed": len(failed),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations)
            }
        else:
            label_stats[label] = {
                "total": len(label_results),
                "successful": 0,
                "failed": len(failed),
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0
            }
    
    # Prepare output
    output_data = {
        "metadata": {
            "mode": "full_capabilities",  # Claude uses all available tools
            "test_mode": "single" if args.test_single else "batch",
            "labels_processed": labels_to_process,
            "batch_start": batch_start.isoformat(),
            "batch_end": batch_end.isoformat(),
            "total_duration_seconds": batch_duration,
            "model": model_name,
            "api_endpoint": "Default Anthropic API",
            "features": ["PubMed plugin", "edge_curation skill", "web search", "extended thinking"]
        },
        "label_statistics": label_stats,
        "runs": all_results
    }
    
    # Create output directory structure (already done above)
    # results_base = PROJECT_ROOT / "results" / "claude_sdk" / "signor_skill_pubmed_search"
    # results_base.mkdir(exist_ok=True)
    
    # Save overall results
    timestamp_str = batch_start.strftime("%Y%m%d_%H%M%S")
    overall_output = results_base / f"batch_results_{timestamp_str}.json"
    with open(overall_output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save results to label-based folders (Moved to inside loop)
    # for label in labels_to_process:
    #     label_dir = results_base / label
    #     label_dir.mkdir(exist_ok=True)
    #     
    #     label_results = [r for r in all_results if r["label"] == label]
    #     
    #     for result in label_results:
    #         # Create filename from edge info
    #         source = result["source"].replace(" ", "_")
    #         target = result["target"].replace(" ", "_")
    #         effect = result["effect"].replace(" ", "_").replace("-", "_")
    #         filename = f"{source}_{target}_{effect}.json"
    #         
    #         output_path = label_dir / filename
    #         with open(output_path, "w") as f:
    #             json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total edges processed: {len(all_results)}")
    print(f"Total time: {batch_duration:.2f}s ({batch_duration/60:.1f} minutes)")
    
    for label, stats in label_stats.items():
        print(f"\n{label.upper()}:")
        print(f"  Total: {stats['total']}")
        print(f"  Successful: {stats['successful']} ✅")
        print(f"  Failed: {stats['failed']}")
        if stats['successful'] > 0:
            print(f"  Avg duration: {stats['avg_duration']:.2f}s")
            print(f"  Min duration: {stats['min_duration']:.2f}s")
            print(f"  Max duration: {stats['max_duration']:.2f}s")
    
    print(f"\nResults saved to:")
    print(f"  Overall: {overall_output}")
    print(f"  By label: {results_base}/[true_positive|true_negative]/")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
