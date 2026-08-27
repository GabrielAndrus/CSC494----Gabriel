import json

import pandas
from tqdm import tqdm
from utils.prompt_utils import TRANSLATE_PROMPT_SYS
from pydantic import BaseModel
from utils.json_fields import FINAL_JSON_FIELDS
from utils.prompt_utils import TRANSLATE_EVALUATE_USR_PROMPT
from utils.process_cultureAtlas_pos import construct_translate_row
from utils.llm_utils import multi_thread_generation_gpt
import argparse

class TRANSLATE_CLASS(BaseModel):
    cultural_group: str
    topic: str
    source: str
    cultural_knowledge: str
    question: str
    answer: str
    grounded_answer: str
    answer_knowledge_points: list[str]
    grounded_answer_knowledge_points: list[str]
    critique_by_points: str
    critique_summary: str
    question_idx: str

def translate(input_path, output_path):
    data = pandas.read_csv(input_path)
    # data = data[:5]
    df_result = []
    messages = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        inputs = {}
        inputs["cultural_group"] = row["cultural_group"]
        inputs["topic"] = row["topic"]
        inputs["source"] = row["source"]
        inputs["cultural_knowledge"] = row["cultural_knowledge"]
        inputs["question"] = row["question"]
        inputs["answer"] = row["answer"]
        inputs["grounded_answer"] = row["grounded_answer"]
        inputs["answer_knowledge_points"] = row["answer_knowledge_points"]
        inputs["grounded_answer_knowledge_points"] = row["grounded_answer_knowledge_points"]
        inputs["critique_by_points"] = row["critique_by_points"]
        inputs["critique_summary"] = row["critique_summary"]
        inputs["question_idx"] = str(idx)

        answer_map = json.loads(row["grounded_answer"])

        languages = answer_map["language"].strip().split(", ")
        for language in languages:
            message = [
                {"role": "user", "content": TRANSLATE_PROMPT_SYS.format(language, json.dumps(inputs))},
            ]
            messages.append(message)

    messages = messages[1501:]
    print(len(messages))
    results = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
    for idx, result in enumerate(results):
        if result is None or result == "":
            continue
        result = json.loads(result)
        df_line = {}
        try:
            if "Error" in str(result):
                continue
            for field in FINAL_JSON_FIELDS:
                df_line[field] = result[field]
        except Exception as e:
            print(result)
            print(f"Error: {e}")
            continue
        df_result.append(df_line)
        if idx % 50 == 0:
            df = pandas.DataFrame(df_result)
            df.to_csv(output_path, encoding='utf-8', index=False)
            print('-'*20 + f'translate Data has been saved to {output_path} at idx {idx}' + '-'*20)
    df = pandas.DataFrame(df_result)
    df.to_csv(output_path, encoding='utf-8', index=False)
    print('-' * 20 + f'Translated Data has been saved to {output_path} of {len(df)} examples' + '-' * 20)


def combine_translate_data(input_path_en, input_path_mul, output_path):
    data_en = pandas.read_csv(input_path_en)
    data_mul = pandas.read_csv(input_path_mul)
    data_mul["data_source"] = [""] * len(data_mul)
    data_en["question_idx"] = [""] * len(data_en)
    for idx, row in tqdm(data_mul.iterrows(), total=len(data_en)):
        try:
            if pandas.isna(row["question_idx"]):
                raise Exception("question_idx is None")
            data_mul.loc[idx, "data_source"] = data_en.loc[int(row["question_idx"]), "data_source"]
        except Exception as e:
            # print(f"Error: {e}")
            data_mul.drop(idx, inplace=True)
            continue
    data_mul.to_csv(input_path_mul, encoding='utf-8', index=False)
    print(len(data_mul))
    data_en.to_csv(input_path_en, encoding='utf-8', index=False)
    data = pandas.concat([data_en, data_mul], axis=1)
    data.to_csv(output_path, encoding='utf-8', index=False)
    print(f"Combined Data has been saved to {output_path}")


def translate_evaluate(input_path_en, input_path_mul, output_path, field="question"):
    data_en = pandas.read_csv(input_path_en)
    data_mul = pandas.read_csv(input_path_mul)
    # data_mul = data_mul[:3]
    columns = data_mul.columns
    rows = []
    messages = []


    for idx, row in tqdm(data_mul.iterrows(), total=len(data_mul)):
        try:
            if field == "question" or "critique_summary":
                translated_answer = row[field]
                ori_answer = data_en.loc[int(row["question_idx"]), field]
            elif field == "answer":
                translated_answer = json.loads(row['answer'])['answer']
                ori_answer = json.loads(data_en.loc[int(row["question_idx"]), "answer"])["answer"]
            else:
                translated_answer = json.loads(row['grounded_answer'])['answer']
                ori_answer = json.loads(data_en.loc[int(row["question_idx"]), "grounded_answer"])["answer"]
            message = [
                {"role": "user", "content": TRANSLATE_EVALUATE_USR_PROMPT.format(ori_answer, translated_answer)},
            ]
            messages.append(message)
            rows.append(construct_translate_row(columns, row))
        except Exception as e:
            # print(row['source'])
            # print(f"Error: {e}")
            # print(row["cultural_knowledge"])
            continue

    data = pandas.DataFrame(rows)
    print(f"There are {len(data)} items, start to translate {len(messages)} messages")
    responses = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
    # print(responses)
    for idx, response in enumerate(responses):
        if response is None:
            responses[idx] == "Yes"
        else:
            if "Yes" in response:
                responses[idx] = "Yes"
            else:
                responses[idx] = "No"

    data['translate_evaluate'] = responses
    data.to_csv(output_path, encoding='utf-8', index=False)
    print(f"Translated Data has been saved to {output_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    translate(args.input_file, args.output_file)
    # You need to adjust your parameter here in translation evaluate
    translate_evaluate(args.input_file, args.output_file, "translate_evaluate.csv", field="question")


