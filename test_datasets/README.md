# Test datasets

Candidate cultural QA datasets for complexity profiling (looking for sufficiently complex data).

## Contents

- candidate1.jsonl - [Dataset name], [link], [N items]
	- Profile: [link to profile JSON]
	- Rationale: [why we're considering this dataset]
- candidate2.jsonl - [Dataset name], [source], [N items]
...

## Schema

Not all datasets are in the same format, so we normalize to the following schema:

{
	"query": "...",
	"location": "...",
	"sub_topic:" "...",
	"ground_truth": {
		"verified_points": ["fact1", "fact2", "fact3", ...]
			}
}
