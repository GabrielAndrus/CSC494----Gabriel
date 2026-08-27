import pandas
from collections import defaultdict
from tqdm import tqdm

def preprocess_data_Atlas(file_path):

    data = pandas.read_csv(file_path)
    examples = {}
    assertion_context_set = defaultdict(set)
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        assertion = row["Assertion"]
        if assertion is None or assertion == "":
            continue
        if assertion not in examples:
            examples[assertion] = defaultdict(str)

        assertion_context_set[assertion].add(row["Context1"] if row["Context1"] != None else "")
        assertion_context_set[assertion].add(row["Context2"] if row["Context2"] != None else "")
        assertion_context_set[assertion].add(row["Context3"] if row["Context3"] != None else "")
        context = "".join(list(assertion_context_set[assertion]))

        examples[assertion]["cultural_group"] = row["Country"]
        examples[assertion]["topic"] = row["Titile"]
        examples[assertion]["context"] += context
        examples[assertion]["url"] = row["Url"]

    result = defaultdict(str)
    for assertion, infos in tqdm(examples.items(), total=len(examples)):
        result['cultural_group'] = infos['cultural_group']
        result['topic'] = infos['topic']
        result['assertion'] = assertion
        result['context'] = infos['context']
        result['source'] = infos['url']

    return result

# def save_file(data, output_file):
#     data
#     data.to_csv(output_file, encoding='utf-8', index=False)
#     print('-'*20 + f'data has been saved to {output_file}' + '-'*20






if __name__ == "__main__":
    file_path = '../datasets/original_data/CultureAtlas-benchmarK_Feb4_pos10k.csv'
    result = preprocess_data_Atlas(file_path)
    print(len(result))
    print(result)

