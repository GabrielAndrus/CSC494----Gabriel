from tqdm import tqdm
import pandas
from utils.llm_utils import multi_thread_generation_gpt
from utils.prompt_utils import (CRITIQUE_PROMPT_SYS,
                                CRITIQUE_PROMPT_USER,
                                CRITIQUE_SUMMARIZATION_PROMPT_USER,
                                CRITIQUE_SUMMARIZATION_PROMPT_SYS)
import json
import argparse

def critique_generation(input_file, output_path):
    data = pandas.read_csv(input_file)
    # data = data[:5]
    messages = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        input = {
            "question": row["question"],
            "grounded_answer": row["grounded_answer"],
            "answer": row["answer"],
            "grounded_knowledge_points": row["grounded_answer_knowledge_points"],
            "knowledge_points_to_critique": row["answer_knowledge_points"]
        }
        message = [
            {"role": "system", "content": CRITIQUE_PROMPT_SYS},
            {"role": "user", "content": CRITIQUE_PROMPT_USER.format(json.dumps(input, indent=4))}
        ]
        messages.append(message)
    results = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
    data["critique_by_points"] = results
    data.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'critique points data has been saved to {output_path}' + '-'*20)

def critique_summarization(input_file, output_file):
    data = pandas.read_csv(input_file)
    # data = data[:5]
    messages = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        critique_point_by_point = row["critique_by_points"]
        message = [
            {"role": "system", "content": CRITIQUE_SUMMARIZATION_PROMPT_SYS},
            {"role": "user", "content": CRITIQUE_SUMMARIZATION_PROMPT_USER.format(json.dumps(critique_point_by_point, indent=4))}
        ]
        messages.append(message)
    results = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
    data["critique_summary"] = results
    data.to_csv(output_file, encoding='utf-8', index=False)
    print('-'*20 + f'critique summary data has been saved to {output_file}' + '-'*20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--input_file_sum', type=str, required=True)
    parser.add_argument('--output_file_sum', type=str, required=True)
    args = parser.parse_args()
    critique_generation(args.input_file, args.output_file)
    critique_summarization(args.inpput_file_sum, args.output_file_sum)


