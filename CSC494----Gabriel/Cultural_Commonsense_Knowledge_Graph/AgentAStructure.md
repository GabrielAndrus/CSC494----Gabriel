# Agent A: Knowledge Graph Representation 

## 1. Structural Representation: Nodes, Edges, Triples

- **Node 1 (Source)**: Represented by the `action` key in the prompt, parsed into the `event` key in Python. This is the initiating behavior or state.  
- **Node 2 (Target)**: Represented by the `knowledge` key. This is the resulting behavior, requirement, or state.  
- **Edge (Predicate)**: Represented by the `relation_type` key, parsed into the `relation` key. The current architecture strictly bounds these to five specific cultural vectors:  
    - `xNext / oNext` (Subsequent actions of the agent/others)
    - `xEffect / oEffect` (Impacts on the agent/others)
    - `xNeed` (Prerequisites for the action)
- **The Triple**: The combination of (`event`, `relation`, `knowledge`).  
- **The Empirical Reasoning Path**: Represented by the `result` (or `event` in the extension phase) key, parsed into `llm_result`. This is the natural-language "If-Then" statement that Agent B will actually read and deliberate over.

## 2. Output Payload Contract

When Agent A executes its retrieval and parsing across the scripts, it standardizes everything into a flat list of dictionaries. For example, if Agent B queries Agent A for cultural context regarding "Indonesia" and "breakfast", the exact empirical contract Agent A delivers looks like this:

```
[
  {
    "event": "buyFromWarungForBreakfast", 
    "knowledge": "experienceLocalCulture", 
    "relation": "xEffect", 
    "llm_result": "If buying breakfast from a warung, then you may experience local culture and community interaction.",
    "location": "Indonesia",
    "sub_topic": "breakfast"
  },
  {
    "event": "wantToAddSpiceToBreakfast",
    "knowledge": "useSambal",
    "relation": "xNext",
    "llm_result": "If wanting to add spice to breakfast, then you might use sambal.",
    "location": "Indonesia",
    "sub_topic": "breakfast"
  }
]
```

## 3. Extension Pathway 

During the `relation_extension` phase, Agent A constructs intermediate steps to form a logical bridge. The payload structure remains the same, but the chronological stacking of the `llm_result` strings is what creates the multi-step cultural reasoning path.

For instance, the payload contract strings together:
- If A, then B. (`intermediate_step`)
- If B, then C. (`intermediate_step`)
- If C, then D. (`next_step`)

This structure is optimized for pluralistic architecture. Agent A provides both the strict graphical triples (`event` $\rightarrow$ `relation` $\rightarrow$ `knowledge`) for deterministic filtering, and the natural language `llm_result` strings for prompt injection.
