import json

import pandas
from tqdm import tqdm
import os
import random
from utils.llm_utils import multi_thread_generation_gpt
import argparse

def postprocessing_cultural_data(input_path, output_path=None):
    data = pandas.read_csv(input_path)
    input_text1 = '''Please answer the following question. Remember to answer in the language of the question\n\nquestion:{}'''
    input_text2 = '''You are an expert in the above cultural group of the question. Please provide a critique of the above answer. The critique should in the language of the answer's language.\n\n '''
    input_text3 = '''Given the above critique, please refine your answer'''
    json_line = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            input_line = {}
            question = row["question"]
            answer = json.loads(row["answer"])
            answer = answer["answer"]
            grounded_answer = json.loads(row["grounded_answer"])
            grounded_answer = grounded_answer["answer"]
            critique = row["critique_summary"]
            input_line["instruction"] = input_text3
            input_line["output"] = grounded_answer
            history = []
            history.append([input_text1.format(question), answer])
            history.append([input_text2, critique])
            input_line["history"] = history
            if pandas.isna(grounded_answer) or pandas.isna(answer) or answer == "" or grounded_answer == ""\
                    or pandas.isna(critique) or critique == "" or question == "" or pandas.isna(question):
                continue
            json_line.append(input_line)
        except Exception as e:
            print(e)
            continue

    print(len(json_line))
    if output_path is None:
        return json_line
    random.shuffle(json_line)
    with open(output_path, 'w') as f:
        json.dump(json_line, f, ensure_ascii=False, indent=4)
    print(f'saving data to {output_path}')


def post_processing_general_data(output_path=None, aya_path=None, mmmlu_path=None, mmlu_path=None, alpaca_path=None):
    # aya
    aya_data = []
    if aya_path is not None:
        instruction_text = f'Please answer the following question, the answer should be in the language of the question'
        data = pandas.read_csv(aya_path)
        data = data.iloc[random.sample(list(range(len(data))), 4000)]
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            instruction = row["inputs"]
            target = row["targets"]
            aya_data.append({"instruction": instruction_text, "input": instruction,"output": target})

    #alpaca
    alpaca_data = []
    if alpaca_path is not None:
        with open(alpaca_path, 'r') as f:
            data = json.load(f)
        alpaca_data = random.sample(data, 4000)
        print(f'alpaca_data length: {len(alpaca_data)}')

    # mmmlu
    mmmlu_data = []
    if mmmlu_path is not None:
        df = pandas.DataFrame()
        instruction_text = f'Answer the following multiple-choice question, you should choose one option among A,B,C,D. You must select one option among "A","B","C","D". Do not output any other things'
        for file_name in os.listdir(mmmlu_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(mmmlu_path, file_name)
                data = pandas.read_csv(file_path)
                df = pandas.concat([df, data])
        df = df.sample(3000)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            question = row["Question"]
            options = [row["A"], row["B"], row["C"], row["D"]]
            options_str = "\n".join([f"{chr(ord('A') + i)} : {op}" for i, op in enumerate(options)])
            question_opt = question + '\n' + options_str
            mmmlu_data.append({"instruction": instruction_text, "input": question_opt, "output": row["Answer"]})


    # mmlu
    mmlu_data = []
    if mmlu_path is not None:
        instruction_text = f'Answer the following multiple-choice question, you should choose one option among A,B,C,D. You must select one option among "A","B","C","D". Do not output any other things'
        df = pandas.DataFrame()
        for file_name in os.listdir(mmlu_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(mmlu_path, file_name)
                data = pandas.read_csv(file_path, header=None)
                df = pandas.concat([df, data])
        df = df.sample(3000)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            question= row[0]
            options = [row[i] for i in range(1, 5)]
            options_str = "\n".join([f"{chr(ord('A') +  i)} : {op}" for i, op in enumerate(options)])
            question_opt = str(question) + '\n' + str(options_str)
            mmlu_data.append({"instruction": instruction_text, "input": question_opt, "output": row[5]})

    final_data = []
    if aya_data is not None:
        final_data.extend(aya_data)
    if alpaca_data is not None:
        final_data.extend(alpaca_data)
    if mmmlu_data is not None:
        final_data.extend(mmmlu_data)
    if mmlu_data is not None:
        final_data.extend(mmlu_data)

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f'saving data to {output_path}')
    else:
        return final_data
    print(f'final_data length: {len(final_data)}')



def mix_data():
    input_path_cultural = "/data/home/frx/cultural_llm/output_data/multilingual_data/process_data/QA_data_translated_final.csv"
    mmmlu_path = '/data/home/frx/cultural_llm/output_data/instruction_following_data/MMMLU/data'
    cultural_data = postprocessing_cultural_data(input_path_cultural)
    instruction_following_data = post_processing_general_data(mmmlu_path=mmmlu_path)
    final_data = []
    final_data.extend(cultural_data)
    final_data.extend(instruction_following_data)
    for data_line in final_data:
        if 'input' not in data_line:
            data_line['input'] = ''
        if 'history' not in data_line:
            data_line['history'] = []
    random.shuffle(final_data)
    output_path = '/data/home/frx/cultural_llm/output_data/multilingual_data/cultural_instruction_dataset_1.json'
    with open(output_path, 'w') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(len(final_data))
    print(f'saving data to {output_path}')


def classify_by_language(input_path, output_path):
    multilingual_data = pandas.read_csv(input_path)
    # multilingual_data = multilingual_data[:10]
    input_text = '''You are given a question and please tell me the language of the question. You should only output the language of the question in one word and your answer should be in English.
    The question is as follows:{}
    Your answer:'''
    messages = []
    for idx, row in tqdm(multilingual_data.iterrows(), total=len(multilingual_data)):
        question = row["question"]
        message = [
            {"role": "user", "content": input_text.format(question)},
        ]
        messages.append(message)
    responses = multi_thread_generation_gpt('gpt-4o-mini', messages=messages, keep_idx=True)
    multilingual_data['language'] = responses
    multilingual_data.to_csv(output_path, index=False)


def postprocessing_cultural_data_ablation(input_path, output_path=None):
    input_text1 = '''Please answer the following question. Remember to answer in the language of the question\n\nquestion:{}'''
    data = pandas.read_csv(input_path)
    json_line = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            input_line = {}
            question = row["question"]
            grounded_answer = json.loads(row["grounded_answer"])
            grounded_answer = grounded_answer["answer"]
            input_line["instruction"] = input_text1.format(question)
            input_line["input"] = ""
            input_line["output"] = grounded_answer
            if pandas.isna(grounded_answer) or grounded_answer == "" or question == "" or pandas.isna(question):
                continue
            json_line.append(input_line)
        except Exception as e:
            print(e)
            continue

    print(len(json_line))
    if output_path is None:
        return json_line
    random.shuffle(json_line)
    with open(output_path, 'w') as f:
        json.dump(json_line, f, ensure_ascii=False, indent=4)
    print(f'saving data to {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    postprocessing_cultural_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()



