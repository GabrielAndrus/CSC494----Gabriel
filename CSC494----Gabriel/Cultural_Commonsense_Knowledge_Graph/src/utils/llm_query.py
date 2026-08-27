import openai 
from openai import OpenAI
from config import OPENAI_API_KEY,HF_TOKEN
from huggingface_hub import login
import re
import time

try:
    import transformers
    import torch
    from vllm import LLM, SamplingParams
except ImportError:
    print("Warning: vLLM or PyTorch not found. Local open-weights models will fail, but GPT-4o will run perfectly.")
    LLM = None
    SamplingParams = None


login(HF_TOKEN)

def clean_tokens(output: str):
    cleaned_output = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', output)
    return cleaned_output.strip()

def query_gpt(prompt, engine='gpt-4o', temp=1, max_retries=3):
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=engine, 
                messages=prompt, 
                temperature=temp,
                response_format={"type": "json_object"} # Crucial for CCKG extraction
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            time.sleep(2 ** attempt) # Exponential backoff
            
    print("Max retries reached. Returning empty JSON.")
    return "{}"


def query_llama(prompt,model_name='meta-llama/Llama-3.1-8B-Instruct',temp=1, model=None, tokenizer=None, llm=None):
    if "meta-llama" in model_name or "google" in model_name :
        formatted_prompts = tokenizer.apply_chat_template(prompt, tokenize=False)
        outputs = llm.generate(formatted_prompts, SamplingParams(temperature=1, max_tokens=2048, top_p=0.95, top_k=-1, seed=0))
        response = clean_tokens(outputs[0].outputs[0].text)
    else:
        formatted_prompts=tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(model.device)
        prompt_padded_len = len(inputs[0])
        gen_tokens = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=2048, do_sample=True,temperature=1, pad_token_id=tokenizer.pad_token_id)
        gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]
        decoded_answer  = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        response=decoded_answer[0]
    
    print('=========NON GPT RESULT========', response)
    return response 
