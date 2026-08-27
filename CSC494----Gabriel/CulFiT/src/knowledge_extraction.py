import pandas
from utils.llm_utils import multi_thread_generation_gpt
from utils.prompt_utils import KNOWLEDGE_PROMPT_SYS_TEMPLATE, KNOWLEDGE_PROMPT_USER_TEMPLATE, KNOWLEGE_EXTRACT_INCONTEXT_DESC,KNOWLEGE_EXTRACT_INCONTEXT_INCONTEXT_EXP
import json
from tqdm import tqdm
import argparse

def knowledge_extraction(file_path, output_path, mode= 'candle'):
    data = pandas.read_csv(file_path)
    # data = data[:5]
    messages = []
    for idx, example in tqdm(data.iterrows(), total=len(data)):
        incontext_user_message = KNOWLEDGE_PROMPT_USER_TEMPLATE.format(KNOWLEGE_EXTRACT_INCONTEXT_INCONTEXT_EXP)
        incontext_assistent_message = KNOWLEGE_EXTRACT_INCONTEXT_DESC
        sys_prompt = KNOWLEDGE_PROMPT_SYS_TEMPLATE
        user_prompt = KNOWLEDGE_PROMPT_USER_TEMPLATE.format(json.dumps(example.to_dict(), indent=4))
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": incontext_user_message},
            {"role": "assistant", "content": incontext_assistent_message},
            {"role": "user", "content": user_prompt}
        ]
        messages.append(message)
    model_name = "gpt-4o-mini"
    response = multi_thread_generation_gpt(model_name, messages, keep_idx=True)
    data['cultural_knowledge'] = response

        # if idx % 100 == 0:
        #     df = pandas.DataFrame(df_results)
        #     df.to_csv(output_path, encoding='utf-8', index=False)
        #     print('-'*20 + f'cultureAtlas Data has been saved to {output_path} at idx' + '-'*20)
    data.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'culture Data has been saved to {output_path}' + '-'*20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()




