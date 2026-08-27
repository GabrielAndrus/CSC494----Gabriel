python pre_process_data.py --input_file './datasets/culturebank_reddit' \
                          --output_file 'Your output_file ' \
                          --model 'culturebank'

python knowledge_extraction.py --input_file 'Your input_file' \
                          --output_file 'Your output_file '

python question_generation.py --input_file 'Your input_file' \
                          --output_file 'Your output_file '

python answer_generation.py --input_file 'Your input_file' \
                          --output_file 'Your output_file ' \

python answer_extract.py --input_file 'Your input_file' \
                          --output_file 'Your output_file ' \

python critique_generation --input_file 'Your input_file' \
                          --output_file 'Your output_file ' \
                          --input_file_sum 'critique summarization input' \
                          --output_file_sum 'critique summarization output'

python translate.py --input_file 'Your input_file' \
                    --output_file 'Your output_file '

python dpo_data_generation.py --input_file 'Your input_file' \
                          --output_file 'Your output_file '

python dataset_post_process.py --input_file 'Your input_file' \
                          --output_file 'Your output_file '
