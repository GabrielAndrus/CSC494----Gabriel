import tenacity
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel


class JSON_KNOWLEDGE(BaseModel):
    # knowledge_points: list[str]
    answer: str
    cultural_group: str
    language: str
    topic: str

@tenacity.retry(stop=tenacity.stop_after_attempt(10))
def openai_response(model_name, messages, temperature=0.7, max_tokens=8192):
    parse=False
    if 'gpt' in model_name.lower():
        client = OpenAI(
            base_url="https://api.chatanywhere.tech/v1",
            api_key= os.environ.get("OPENAI_API_KEY", "0")
        )
        parse = True



    else:
        client = OpenAI(
            base_url="http://0.0.0.0:8001/v1",
            api_key='0'
        )
    try:
        if parse:
            response = client.beta.chat.completions.parse(

                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                seed=1234,
                # response_format=JSON_KNOWLEDGE
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                seed=1234,
            )
        result = response.choices[0].message.content.strip()
        if result.startswith("Error:"):
            raise Exception(f"Error: {result}")
        return result
    except Exception as e:
        print(f"Error: {e}")


@tenacity.retry(stop=tenacity.stop_after_attempt(10))
def lama_generation(model, tokenizer, input_messages=None, temperature=0.7, max_tokens=4096):
    try:
        # prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
        # inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        inputs = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt = inputs["input_ids"][0]
        # prompt_decode = tokenizer.decode(prompt, skip_special_tokens=False)
        # print(prompt_decode)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        outputs = tokenizer.decode(outputs['sequences'][0][len(prompt):], skip_special_tokens=True)
        # outputs = outputs[(outputs.rfind("[/INST]") + len("[/INST]")):]
        return outputs

    except Exception as e:
        print(f"Error encountered while processing data: {e}")


def multi_thread_generation_gpt(model_name, messages, keep_idx=False, temperature=0.7, max_tokens=8192):
    results = []
    if keep_idx:
        results = [None] * len(messages)
    with tqdm(total=len(messages)) as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(openai_response, model_name, message, temperature, max_tokens): idx for idx, message in enumerate(messages)}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if keep_idx:
                        idx = futures[future]
                        results[idx] = result
                    else:
                        results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
    return results





if __name__ == "__main__":
    model_name = "gpt-4o"

    messages = []
    for i in tqdm(range(10), total=10):
        message = [
            {"role": "user", "content": f'what is 1 + {i}'},
        ]
        messages.append(message)
    response = multi_thread_generation_gpt(model_name, messages, keep_idx=True)
    print(response)
