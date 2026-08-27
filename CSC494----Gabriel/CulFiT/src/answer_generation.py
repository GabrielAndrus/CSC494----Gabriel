from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, PeftModel
from utils.llm_utils import lama_generation, openai_response, multi_thread_generation_gpt
from utils.prompt_utils import (ANSER_GENERATION_USER_PROMT, ANSWER_GENERATION_SYS_PROMPT,
                                ANSWER_GENERATION_IN_CONTEXT_EXAMPLE, ANSWER_GENERATION_INCONTXT_QUESTION,
                                ANSWER_GENERATION_USER_PROMPT_GPT, ANSWER_GENERATION_SYS_PROMPT_GPT,
                                ANSWER_GENERATION_INCONTEXT_KNOWLEDGE)
import pandas
import argparse

def generate_answer_llama(model, tokenizer, input_path, output_path,adapter_path=None):
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    data = pandas.read_csv(input_path)
    # data = data[:5]
    print(f'There are {len(data)} examples in the dataset')
    answers = []
    df = []
    messages = []
    if adapter_path is not None:
        lora_model = PeftModel.from_pretrained(model, adapter_path)
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        question = row["question"]
        incontext_user_message = ANSER_GENERATION_USER_PROMT.format(ANSWER_GENERATION_INCONTXT_QUESTION)
        message = [
            {"role": "system", "content": ANSWER_GENERATION_SYS_PROMPT},
            {"role": "user", "content": incontext_user_message},
            {"role": "assistant", "content": ANSWER_GENERATION_IN_CONTEXT_EXAMPLE},
            {"role": "user", "content": ANSER_GENERATION_USER_PROMT.format(question)}
        ]
        if adapter_path is not None:
            answer = lama_generation(lora_model, tokenizer, message)
            print(answer)
        else:
            answer = lama_generation(model, tokenizer, message)
        # answer = answer[answer.find("{"):]
        answers.append(answer)
        df.append(row)
        if idx % 20 == 0:
            df_results = pandas.DataFrame(df)
            df_results.to_csv(output_path, encoding='utf-8', index=False)
            print('-' * 20 + f'culture Data has been saved to {output_path}' + '-' * 20)

    data["answer"] = answers
    data.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'culture Data has been saved to {output_path}' + '-'*20)


def generate_grounded_answer_gpt(input_path, output_path, model_name='gpt-4o-mini', mode='grounded'):
    data = pandas.read_csv(input_path)
    # data = data[:10]
    print(f'There are {len(data)} examples in the dataset')
    messages = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        question = row["question"]
        incontext_user_message = ANSWER_GENERATION_USER_PROMPT_GPT.format(ANSWER_GENERATION_INCONTXT_QUESTION, ANSWER_GENERATION_INCONTEXT_KNOWLEDGE)
        if mode == 'grounded':
            message = [
                {"role": "system", "content": ANSWER_GENERATION_SYS_PROMPT_GPT},
                {"role": "user", "content": incontext_user_message},
                {"role": "assistant", "content": ANSWER_GENERATION_IN_CONTEXT_EXAMPLE},
                {"role": "user", "content": ANSWER_GENERATION_USER_PROMPT_GPT.format(question, row["cultural_knowledge"])}
            ]
        else:
            message = [
                {"role": "system", "content": ANSWER_GENERATION_SYS_PROMPT},
                {"role": "user", "content": incontext_user_message},
                {"role": "assistant", "content": ANSWER_GENERATION_IN_CONTEXT_EXAMPLE},
                {"role": "user", "content": ANSER_GENERATION_USER_PROMT.format(question)}
            ]
        messages.append(message)
    # model_name = "gpt-4o-mini"
    response = multi_thread_generation_gpt(model_name, messages, keep_idx=True)

    if mode == 'grounded':
        data["grounded_answer"] = response
    else:
        data["answer"] = response
    data.to_csv(output_path, encoding='utf-8', index=False)
    print('-'*20 + f'Data has been saved to {output_path}' + '-'*20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_llama', type=str, required=True)
    parser.add_argument('--output_file_llama', type=str, required=True)
    parser.add_argument('--input_file_gpt', type=str, required=True)
    parser.add_argument('--output_file_gpt', type=str, required=True)
    args = parser.parse_args()

    # model_name = '/models/Meta-Llama-3.1-8B-Instruct'
    generate_grounded_answer_gpt(args.input_file_gpt, args.output_file_gpt, mode='grounded')


