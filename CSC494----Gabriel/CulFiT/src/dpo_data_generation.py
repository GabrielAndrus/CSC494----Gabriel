from utils.llm_utils import multi_thread_generation_gpt
import pandas
from tqdm import tqdm
from utils.prompt_utils import KNOWLEDGE_ENTAIL_SYSTEM_PROMPT, KNOWLEDGE_ENTAIL_USER_TEMPLATE, REFINE_PROMPT_USER
import json
import random
import argparse

def is_entailment(input_path, output_path):
    data = pandas.read_csv(input_path)
    # data = data[:10]
    messages = []
    print(f'There are {len(data)} examples in the dataset')
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        answer = json.loads(row["grounded_answer"])["answer"]
        cultural_knowledge = row["cultural_knowledge"]
        user_message = KNOWLEDGE_ENTAIL_USER_TEMPLATE.format(answer, cultural_knowledge)
        message = [
            {"role": "system", "content": KNOWLEDGE_ENTAIL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        messages.append(message)
    responses = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
    data["entailment"] = responses
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        if str(row['entailment']) == "No":
            data.drop(idx, inplace=True)
    data.to_csv(output_path, encoding='utf-8', index=False)
    print(f'There are {len(data)} entailmented examples after filtering')
    print('-'*20 + f'entailment Data has been saved to {output_path}' + '-'*20)

    count = 0
    for response in responses:
        if str(response).lower() == "no":
            count += 1
    print(f'There are {count} unentailmented responses out of {len(responses)}')


def pure_data_generation(input_path, output_path):
    json_lines = []
    instruction = '''You are a helpful assistant for a culturally answering scenario. Your goal is to provide a culturally appropriate answer to the user's question.'''
    data = pandas.read_csv(input_path)
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            input_line = {}
            if row["question"] is None or row["answer"] is None or row["grounded_answer"] is None:
                continue
            question = row["question"]
            input_line["instruction"] = instruction
            input_line["input"] = str(question)
            input_line["chosen"] = str(json.loads(row["grounded_answer"])["answer"])
            input_line["rejected"] = str(json.loads(row["answer"])["answer"])
            json_lines.append(input_line)
        except Exception as e:
            print(f'Error: {e}')
            continue
    random.shuffle(json_lines)
    print(f'There are {len(json_lines)} examples in the dataset')
    with open(output_path, 'w') as f:
        json.dump(json_lines, f, ensure_ascii=False, indent=4)
    print(f'saving data to {output_path}')


def refine_data_generation(input_path, output_path, model_name='llama3.1'):
    messages = []
    data = pandas.read_csv(input_path)
    # data = data[:5]
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            question = row["question"]
            critique_summary = row["critique_summary"]
            answer = json.loads(row["answer"])["answer"]
            message = [{
                "role": "user",
                "content": REFINE_PROMPT_USER.format(question, answer, critique_summary)
            }]
            messages.append(message)
        except Exception as e:
            data.drop(idx, inplace=True)
            print(f'Error: {e}')
            continue
    response = multi_thread_generation_gpt(model_name, messages, keep_idx=True)
    data["refined_answer"] = response
    data.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'Data has been saved to {output_path}' + '-'*20)


def prune_eval_data(input_path, output_path, f1_threshold=0.7):
    data = pandas.read_csv(input_path)
    print(f'There are {len(data)} examples in the dataset')
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        if float(row["ori_f1_score"]) > f1_threshold:
            data.drop(idx, inplace=True)
    print(f'There are {len(data)} examples in the dataset after pruning')
    data.to_csv(output_path, encoding='utf-8', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    # is_entailment(input_path, output_path)
    # pure_data_generation(input_path, output_path)
    refine_data_generation(args.input_file, args.output_file)
    # prune_eval_data(input_path, output_path, f1_threshold=0.7)


