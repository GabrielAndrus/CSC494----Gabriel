import pandas
import openai
import tenacity
from collections import defaultdict

def preprocess_data_Atlas(file_path):
    data = pandas.read_csv(file_path)
    data = data[:1]
    examples = {}
    assertion_context_set = defaultdict(set)
    for idx, row in data.iterrows():
        assertion = row["Assertion"]
        if assertion == None:
            continue
        if assertion not in examples:
            examples[assertion] = defaultdict(str)

        assertion_context_set[assertion].add(row["Context1"] if row["Context1"] != None else "")
        assertion_context_set[assertion].add(row["Context2"] if row["Context2"] != None else "")
        assertion_context_set[assertion].add(row["Context3"] if row["Context3"] != None else "")
        context = "".join(list(assertion_context_set[assertion]))


        examples[assertion]["context"] += context
        examples[assertion]["country"] = row["Country"]
        examples[assertion]["title"] = row["Titile"]
        examples[assertion]["url"] = row["Url"]
    print(examples)
    return examples

def construct_row(row, data_source=None):
    output = {}
    output["cultural_group"] = row["cultural_group"]
    output["topic"] = row["topic"]
    output["source"] = row["source"]
    output["cultural_knowledge"] = row["cultural_knowledge"]
    output["question"] = row["question"]
    output["answer"] = row["answer"]
    output["grounded_answer"] = row["grounded_answer"]
    output["answer_knowledge_points"] = row["answer_knowledge_points"]
    output["grounded_answer_knowledge_points"] = row["grounded_answer_knowledge_points"]
    output["critique_by_points"] = row["critique_by_points"]
    output["critique_summary"] = row["critique_summary"]
    output["question_idx"] = row["question_idx"]
    # if row["data_source"] is not None:
    #     output["data_source"] = row["data_source"]

    if data_source:
        output["data_source"] = data_source
    return output

def construct_culturebank_row(row, idx=None):
    output = {}
    output["cultural_group"] = row["cultural_group"]
    output["topic"] = row["topic"]
    output["cultural_knowledge"] = row["cultural_knowledge"]
    output["question"] = row["question"]
    output["answer"] = row["answer"]
    output["grounded_answer"] = row["grounded_answer"]
    output["answer_knowledge_points"] = row["answer_knowledge_points"]
    output["grounded_answer_knowledge_points"] = row["grounded_answer_knowledge_points"]
    if idx is not None:
        output["idx"] = idx
    return output


def construct_translate_row(columns, row):
    output = {}
    for column in columns:
        output[column] = row[column]
    return output

if __name__ == "__main__":
    file_path = './datasets/CultureAtlas-benchmarK_Feb4_pos10k.csv'
    examples = preprocess_data_Atlas(file_path)
