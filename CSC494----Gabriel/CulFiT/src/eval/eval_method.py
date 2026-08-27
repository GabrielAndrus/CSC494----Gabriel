
import json
import pandas
from tqdm import tqdm
from utils.llm_utils import multi_thread_generation_gpt
from utils.prompt_utils import (
    EVAL_CULTURAL_POINTS_PROMPT,
    EVAL_CULTURAL_GROUP_PROMPT,
    EVAL_TOPIC_PROMPT,
    EVAL_CULTURAL_LANGUAGE_PROMPT
)
from utils.process_cultureAtlas_pos import construct_row, construct_culturebank_row
import argparse


class EvalClass:
    def __init__(self, answer_topic, answer_languages, answer_cultural_group, answer_knowledge_points, grounded_answer_knowledge_points,
                 grounded_topic, grounded_answer, grounded_cultural_group, grounded_languages, input_file=None, output_file=None):
        self.answer_topic = answer_topic
        self.answer_languages = answer_languages
        self.answer_cultural_group = answer_cultural_group
        self.answer_knowledge_points = answer_knowledge_points
        self.grounded_answer_knowledge_points = grounded_answer_knowledge_points
        self.grounded_topic = grounded_topic
        self.grounded_answer = grounded_answer
        self.grounded_cultural_group = grounded_cultural_group
        self.grounded_languages = grounded_languages


    def get_messages (self, mode="precision"):
        precision_messages = []
        message_cultural_group = [{
            "role": "user",
            "content": EVAL_CULTURAL_GROUP_PROMPT.format(self.answer_cultural_group,self.grounded_cultural_group)
        }]
        message_topic = [{
            "role": "user",
            "content": EVAL_TOPIC_PROMPT.format(self.answer_topic,self.grounded_topic)
        }]
        for language in self.answer_languages:
            message_language = [{
                "role": "user",
                "content": EVAL_CULTURAL_LANGUAGE_PROMPT.format(language, self.grounded_languages)
            }]
            precision_messages.append(message_language)
        if mode == "precision":
            for knowledge_point in self.answer_knowledge_points:
                message_knowledge_point = [{
                    "role": "user",
                    "content": EVAL_CULTURAL_POINTS_PROMPT.format(knowledge_point, self.grounded_answer_knowledge_points)
                }]
                precision_messages.append(message_knowledge_point)
        elif mode == "recall":
            for grounded_knowledge_point in self.grounded_answer_knowledge_points:
                message_knowledge_point = [{
                    "role": "user",
                    "content": EVAL_CULTURAL_POINTS_PROMPT.format(grounded_knowledge_point, self.answer_knowledge_points)
                }]
                precision_messages.append(message_knowledge_point)
        else:
            raise ValueError("Invalid mode")
        precision_messages.append(message_cultural_group)
        precision_messages.append(message_topic)
        return precision_messages


    def extract_answer(self, response):
        new_response = []
        for res in response:
            if res is not None:
                if "Yes" in res:
                    new_response.append("Yes")
                else:
                    new_response.append("No")
            else:
                new_response.append("No")
                continue
        print(new_response)
        print(response)
        return new_response

    def cal_precision(self):
        total_length = len(self.answer_knowledge_points) + len(self.answer_languages) + 2
        messages = self.get_messages("precision")
        results = multi_thread_generation_gpt("gpt-4o-mini", messages, keep_idx=True)
        results = self.extract_answer(results)
        # for language in self.answer_languages:
        #     if language in self.grounded_languages:
        #         results.append("Yes")
        #     else:
        #         results.append("No")
        return sum([1 for result in results if result == "Yes"]) / total_length

    def cal_recall(self):
        total_length = len(self.grounded_answer_knowledge_points) + len(self.grounded_languages) + 2
        results = multi_thread_generation_gpt("gpt-4o-mini", self.get_messages("recall"), keep_idx=True)
        results = self.extract_answer(results)
        # for language in self.grounded_languages:
        #     if language in self.answer_languages:
        #         results.append("Yes")
        #     else:
        #         results.append("No")
        return sum([1 for result in results if result == "Yes"]) / total_length

    def calculate_f1_score(self, precision, recall):
        sigma = 1e-6
        return 2 * (precision * recall) / (precision + recall + sigma)


def saving_file(rows, precision_all, recall_all, f1_score_all, output_file):
    df = pandas.DataFrame(rows)
    eval_length = len(df)
    df["precision"] = precision_all
    df["recall"] = recall_all
    df["f1_score"] = f1_score_all
    precision_avg = sum(precision_all) / eval_length
    recall_avg = sum(recall_all) / eval_length
    f1_score_avg = sum(f1_score_all) / eval_length
    print(f"Precision avg: {precision_avg}, Recall avg: {recall_avg}, F1 Score avg: {f1_score_avg}")
    df.to_csv(output_file, encoding='utf-8', index=False)
    print('-'*20 + f'evaluation data has been saved to {output_file}' + '-'*20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    data = pandas.read_csv(args.input_file)
    precision_all = []
    recall_all = []
    f1_score_all = []
    rows = []


    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            answer = json.loads(row["answer"])
            answer_topic = answer["topic"]
            answer_languages = answer["language"].strip().split(', ')
            answer_cultural_group = answer["cultural_group"]
            answer_knowledge_points = list(json.loads(row["answer_knowledge_points"])["knowledge_points"])
            grounded_answer_knowledge_points = list(json.loads(row["grounded_answer_knowledge_points"])["knowledge_points"])
            grounded_topic = row["topic"]
            grounded_answer = json.loads(row["grounded_answer"])
            grounded_cultural_group = row["cultural_group"]
            grounded_languages = grounded_answer["language"].strip().split(', ')
            eval_class = EvalClass(answer_topic, answer_languages, answer_cultural_group,
                                   answer_knowledge_points, grounded_answer_knowledge_points, grounded_topic,
                                   grounded_answer, grounded_cultural_group, grounded_languages)
            precision = eval_class.cal_precision()
            recall = eval_class.cal_recall()
            f1_score = eval_class.calculate_f1_score(precision, recall)

            precision_all.append(precision)
            recall_all.append(recall)
            f1_score_all.append(f1_score)
            rows.append(construct_culturebank_row(row))
            print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
            if idx % 20 == 0:
                saving_file(rows, precision_all, recall_all, f1_score_all, args.output_file)
                print('-'*20 + f'evaluation data has been saved to {args.output_file} at idx {idx}' + '-'*20)
        except Exception as e:
            print(f'Error at idx {idx}: {e}')
            saving_file(rows, precision_all, recall_all, f1_score_all, args.output_file)
            continue

    saving_file(rows, precision_all, recall_all, f1_score_all, args.output_file)




if __name__ == "__main__":
    main()



