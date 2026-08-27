# Agent B: Connecting with Agent A

## Inter-agent Contract 

For CulFit to act as Agent B, we must take note of its evaluation framework, which requires models to evaluate text across three specific contextual units: 
- cultural group affiliation
- cultural topic
- primary language(s) of the cultural group
 
We can map Agent A's existing JSON payload directly to Agent B's requirements:
- Agent A's `location` $\rightarrow$ CulFiT's cultural group affiliation and primary language(s).
- Agent A's `sub_topic` $\rightarrow$ CulFiT's cultural topic.
- Agent A's `llm_result` (The reasoning path) $\rightarrow$ raw cultural text that Agent B will decompose into verifiable knowledge units.
