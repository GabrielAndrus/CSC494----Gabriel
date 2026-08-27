# CulFiT

**Official code implementation for ACL25 'CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis'**
![main](./imgs/main.jpg)


# Inference

we use [vllm](https://github.com/vllm-project/vllm) for fast inference

You can serve your vllm engin on 8000 port using

```shell
vllm serve "Qwen2.5" --host 0.0.0.0 --port 8080
```

# Data Generation

**Preparation**

all source data can be downloaded through https://drive.google.com/drive/folders/1gOoU8KXUiUASguRuBr0bJPVyYTj_t5Wh?usp=drive_link

**Generation**

We also provide the scripts for all our training data synthesis

```shell
bash ./src/scripts.sh
```

#Evaluation
You can evaluate your results by using
```python
python ./src/eval/eval_method.py --input_file YOUR_INPUT_FILE --output_file YOUR_OUTPUT_FILE
```

# Acknowledgements

All our training processes are based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), training scripts can be found in LLaMA-Factory/llama3_lora_sft_ds3.yaml



# Citation

coming soon
