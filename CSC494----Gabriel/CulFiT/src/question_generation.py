import pandas
from tqdm import tqdm
from utils.llm_utils import multi_thread_generation_gpt
from utils.prompt_utils import  QUESTION_GENERATION_SYS_PROMPT, QUESTION_GENERATION_USER_PROMPT
import argparse

def question_generation(file_path, output_path):
    data = pandas.read_csv(file_path)
    # data = data[:10]
    messages = []
    rows = []
    for idx, row in tqdm(data.iterrows(), total = len(data)):
        sys_message = QUESTION_GENERATION_SYS_PROMPT
        input_dict = {"cultural_group": row["cultural_group"], "topic": row["topic"], "source": row["url"], "cultural_knowledge": row["cultural_knowledge"]}
        user_message = QUESTION_GENERATION_USER_PROMPT.format(input_dict, incindent=4)
        message = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": user_message}
        ]
        messages.append(message)
        rows.append({"cultural_group": row["cultural_group"], "topic": row["topic"], "source": row["url"], "cultural_knowledge": row["cultural_knowledge"]})
    model_name = "gpt-4o-mini"
    response = multi_thread_generation_gpt(model_name, messages, keep_idx=True)
    df = pandas.DataFrame(rows)
    df['question'] = response
    df.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'culture Data has been saved to {output_path}' + '-'*20)

        # if idx % 100 == 0:
        #     df = pandas.DataFrame(df_results)
        #     df.to_csv(output_path, encoding='utf-8', index=False)
        #     print('-'*20 + f'cultureAtlas Data has been saved to {output_path} at idx {idx}' + '-'*20)
    # df_results = pandas.DataFrame(df_results)
    # df_results.to_csv(output_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    question_generation(args.input_file, args.output_file)
