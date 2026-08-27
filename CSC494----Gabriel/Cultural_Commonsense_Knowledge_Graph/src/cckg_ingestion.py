import json
import logging
from datetime import datetime
import openai # or your preferred LLM provider

# 3. Establish the Metric: Logging Setup
log_filename = f"extraction_logs/cckg_run_{datetime.now().strftime('%Y%m%d')}.jsonl"

def log_attempt(culture, query, prompt, reasoning, output, error=None):
    """Logs the exact inputs, reasoning paths, and outputs/errors to JSONL."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "culture": culture,
        "query": query,
        "prompt": prompt,
        "raw_reasoning": reasoning,
        "extracted_graph": output,
        "error": str(error) if error else None
    }
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def extract_cultural_nodes(culture, context_query):
    """Step 1: Elicit culture-specific entities and practices."""
    prompt = f"Identify 3-5 core cultural entities, values, or practices in {culture} related to: {context_query}. Return as a JSON list of strings."
    
    # Example API call (adapt to your specific client/local model)
    response = openai.chat.completions.create(
        model="gpt-4-turbo", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def build_inferential_edges(culture, nodes):
    """Step 2: Compose multi-step inferential chains (edges) between nodes."""
    prompt = f"Given these cultural concepts for {culture}: {nodes}. Construct an inferential chain (if-then relations) connecting them. Return as JSON: {{'edges': [{'source': 'X', 'relation': 'leads_to', 'target': 'Y', 'reasoning': '...'}]}}"
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content, json.loads(response.choices[0].message.content)

def run_ingestion_pipeline(culture, query):
    try:
        # Iterative Framework
        nodes_dict = extract_cultural_nodes(culture, query)
        raw_reasoning, edges_dict = build_inferential_edges(culture, nodes_dict)
        
        graph = {"nodes": nodes_dict, "edges": edges_dict}
        log_attempt(culture, query, "Iterative Prompts", raw_reasoning, graph)
        print(f"Success. Logged {len(edges_dict.get('edges', []))} edges.")
        return graph
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        log_attempt(culture, query, "Iterative Prompts", None, None, error=e)

# Example Execution
# run_ingestion_pipeline("Morocco", "hospitality and guest welcoming")