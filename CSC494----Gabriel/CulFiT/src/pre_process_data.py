

import pandas
from collections import defaultdict
import random
import argparse
from tqdm import tqdm

class ProcessData:
    def __init__(self, input_path, output_path, mode='culturebank'):
        self.input_path = input_path
        self.output_path = output_path
        self.mode = mode
        self.data = None

    def _preprocess_data_Atlas(self):
        data = pandas.read_csv(self.input_path)
        # data = data[:]
        examples = {}
        assertion_context_set = defaultdict(set)
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            assertion = row["Assertion"]
            if assertion is None or assertion == "" or pandas.isna(assertion):
                continue

            # check if the assertion prefix is in the dictionary
            for key, _ in list(examples.items()):
                if str(assertion).startswith(str(key)):
                    examples.pop(key)

            if assertion not in examples:
                examples[assertion] = defaultdict(str)

            assertion_context_set[assertion].add(row["Context1"] if row["Context1"] != None else "")
            assertion_context_set[assertion].add(row["Context2"] if row["Context2"] != None else "")
            assertion_context_set[assertion].add(row["Context3"] if row["Context3"] != None else "")
            context = "".join(list(assertion_context_set[assertion]))


            examples[assertion]["context"] += context
            examples[assertion]["cultural_group"] = row["Country"]
            examples[assertion]["topic"] = row["Titile"]
            examples[assertion]["url"] = row["Url"]
        results = []
        for assertion, value in examples.items():
            results.append({"cultural_group": value["cultural_group"], "topic": value["topic"], "assertion": assertion,
                            "context": value["context"], "url": value["url"]})
        return results



    def _pre_process_data_candle(self):
        data = pandas.read_json(self.input_path, lines=True)
        examples = []
        country_indices, religion_indices, continent_indices = self._get_candle_idx()
        all_indices = country_indices + religion_indices + continent_indices
        data = data.iloc[all_indices]
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            raw_sentences = row['raw_sentences']
            context = ""
            urls = ""
            for sentence_url in raw_sentences:
                context += sentence_url['text'] + '\n'
                urls += sentence_url['url'] + ', '
            topic = f'{row["facet"]} of {row["subject"]}'
            examples.append({
                "cultural_group": row['subject'],
                "topic": topic,
                "assertion": row['assertion'],
                "context": context,
                "url": urls
            })
        print(f'candle data: {len(examples)}')
        return examples

    def _get_candle_idx(self, max_count=8000):
        data = pandas.read_json(self.input_path, lines=True)
        country_distribution = defaultdict(list)
        religion_distribution = defaultdict(list)
        continent_distribution = defaultdict(list)
        country_indices = []
        religion_indices = []
        continent_indices = []
        for idx, row in enumerate(data.iterrows()):
            if float(row[1]['combined_score']) >= 0.5:
                if row[1]['domain'] == 'countries':
                    country_distribution[row[1]['subject']].append(idx)
                elif row[1]['domain'] == 'religions':
                    religion_distribution[row[1]['subject']].append(idx)
                elif row[1]['domain'] == 'continents':
                    continent_distribution[row[1]['subject']].append(idx)

        country_distribution = {k : v for k , v in country_distribution.items() if len(v) >= 15}  # 原先 900 + 1500 + 5000 （3， 6， 3）
        for country, country_idx in country_distribution.items(): #
            country_indices.extend(random.sample(country_idx, k=len(country_idx)//2))
        for religion, religion_idx in religion_distribution.items():
            religion_indices.extend(random.sample(religion_idx, k=len(religion_idx)//3))
        for continent, continent_idx in continent_distribution.items():
            continent_indices.extend(random.sample(continent_idx, k=len(continent_idx)//2))

        print(f'country_indices: {len(country_indices)}, religion_indices: {len(religion_indices)}, continent_indices: {len(continent_indices)}')
        return country_indices, religion_indices, continent_indices




    def _pre_process_data_CulturalBank(self):
        data = pandas.read_csv(self.input_path)
        examples = []

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            if float(row["agreement"]) < 0.6:
                continue
            examples.append({
                "cultural_group": row["cultural_group"],
                "topic":  row["cultural_group"] + "'s " + row["topic"],
                "url": "https://www.reddit.com",
                "cultural_knowledge": row['eval_whole_desc'],
                "question": row["eval_question"],
            })

        return examples

    def _saving_data(self, output_path, data, mode='culturebank'):
        data = pandas.DataFrame(data)
        data.to_csv(output_path, encoding='utf-8', index=False)
        print('-'*20 + f'{mode} data has been saved to {output_path}' + '-'*20)

    def process_data(self):
        if self.mode == 'culturebank':
            data = self._pre_process_data_CulturalBank()
        elif self.mode == 'atlas':
            data = self._preprocess_data_Atlas()
        elif self.mode == 'candle':
            data = self._pre_process_data_candle()
        else:
            print('Invalid mode')
        self.data = data
        if self.data is not None:
            self._saving_data(self.output_path, data, self.mode)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--mode', type=str, default='culturebank', choices=['culturebank', 'atlas', 'candle'],
                        help='Mode of processing data')
    args = parser.parse_args()
    processing_data = ProcessData(args.input_file, args.output_file, args.mode)
    processing_data.process_data()
