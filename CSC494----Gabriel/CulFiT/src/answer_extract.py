import json
import argparse

from tqdm import tqdm
import pandas
from utils.prompt_utils import ANSWER_EXTRACT_PROMPT_SYS, ANSWER_EXTRACT_PROMPT_USER
from utils.llm_utils import multi_thread_generation_gpt, openai_response

def answer_extract(input_file, output_file):
    data = pandas.read_csv(input_file, encoding="utf-8", encoding_errors="ignore")
    # data = data[:10]
    answer_messages = []
    grounded_answer_messages = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        answer = row["answer"]
        grounded_answer = row["grounded_answer"]
        question = row['question']
        if answer == "" or grounded_answer == "":
            continue
        answer_input = {"question": question, "answer": answer}
        grounded_answer_input = {"question": question, "answer": grounded_answer}
        answer_message = [
            {"role": "system", "content": ANSWER_EXTRACT_PROMPT_SYS},
            {"role": "user", "content": ANSWER_EXTRACT_PROMPT_USER.format(json.dumps(answer_input, indent=4))}
        ]
        grounded_answer_message = [
            {"role": "system", "content": ANSWER_EXTRACT_PROMPT_SYS},
            {"role": "user", "content": ANSWER_EXTRACT_PROMPT_USER.format(json.dumps(grounded_answer_input, indent=4))}
        ]
        answer_messages.append(answer_message)
        grounded_answer_messages.append(grounded_answer_message)

    answer_knowledge_points = multi_thread_generation_gpt("gpt-4o-mini", answer_messages, keep_idx=True)
    # grounded_answer_knowledge_points = multi_thread_generation_gpt("gpt-4o-mini", grounded_answer_messages, keep_idx=True)
    data["answer_knowledge_points"] = answer_knowledge_points
    # data["grounded_answer_knowledge_points"] = grounded_answer_knowledge_points
    data.to_csv(output_file, encoding='utf-8', index=False)
    print('-'*20 + f'knowledge points data has been saved to {output_file}' + '-'*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    answer_extract(args.input_file, args.output_file)

