import pandas

from VSM13_dict import vsm13_data, country_dict, vsm13_data_question, get_option_str
import re, math
import codecs, csv
import json
from utils.llm_utils import openai_response, multi_thread_generation_gpt
import argparse

def getResponse(prompts, engine, culture):
    msgs = []

    for idx, prompt in enumerate(prompts):
        option_str = get_option_str(idx)

        input_text = f'''To answer the following multiple-choice question, you should choose one option among A,B,C,D,E. 
        Question: {prompt}  
        {option_str}

        Your answer:
        '''

        msg = [
            {"role": "system", "content": f"You are a {culture} chatbot that know {culture} very well."},
            {"role": "user", "content": input_text}
        ]
        msgs.append(msg)


    responses = multi_thread_generation_gpt(engine, msgs, keep_idx=True)
    return responses


def computeMetrics(ans_list):
    pdi = 35 * (int(ans_list[7 - 1]) - int(ans_list[2 - 1])) + 25 * (int(ans_list[20 - 1]) - int(ans_list[23 - 1]))
    idv = 35 * (int(ans_list[4 - 1]) - int(ans_list[1 - 1])) + 35 * (int(ans_list[9 - 1]) - int(ans_list[6 - 1]))
    mas = 35 * (int(ans_list[5 - 1]) - int(ans_list[3 - 1])) + 25 * (int(ans_list[8 - 1]) - int(ans_list[10 - 1]))
    uai = 40 * (int(ans_list[18 - 1]) - int(ans_list[15 - 1])) + 25 * (int(ans_list[21 - 1]) - int(ans_list[24 - 1]))
    lto = 40 * (int(ans_list[13 - 1]) - int(ans_list[14 - 1])) + 25 * (int(ans_list[19 - 1]) - int(ans_list[22 - 1]))
    ivr = 35 * (int(ans_list[12 - 1]) - int(ans_list[11 - 1])) + 40 * (int(ans_list[17 - 1]) - int(ans_list[16 - 1]))

    return pdi + 50, idv + 50, mas + 50, uai + 50, lto + 50, ivr + 50

def get_number(response):
    if response is None or response == '':
        return 10 # no answer
    return ord(response) - ord('A') + 1

def answer_extract(response, row=None):
    choices = ['A', 'B', 'C', 'D', 'E']
    answer = ""
    for choice in choices:
        if choice in response:
            answer = choice
            break
    return answer


def run(culture, output_path=None, engine=None):
    ans_dict = dict()
    with codecs.open(f'data/6-dimensions-for-website-2015-08-16.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True, delimiter=';'):
            country = row['country']
            pdi = row['pdi']
            idv = row['idv']
            mas = row['mas']
            uai = row['uai']
            lto = row['ltowvs']
            ivr = row['ivr']
            ans_dict[country] = {'pdi': pdi, 'idv': idv, 'mas': mas, 'uai': uai, 'lto': lto, 'ivr': ivr}

    cur_country = country_dict[culture]
    human_ans = ans_dict[cur_country]
    print('Human', human_ans)

    questions = [vsm13_data_question[key] for key in vsm13_data_question.keys()]
    responses = getResponse(questions, engine, culture)
    ans_list = [answer_extract(response) for response in responses]
    print(f'Ans list: {ans_list}')
    ans_list = [get_number(ans) for ans in ans_list]
    pdi, idv, mas, uai, lto, ivr = computeMetrics(ans_list)
    cur_ans = {'pdi': pdi, 'idv': idv, 'mas': mas, 'uai': uai, 'lto': lto, 'ivr': ivr}
    print('Cur', pdi, idv, mas, uai, lto, ivr)

    missed_key = []
    human_point = []
    cur_point = []
    for key in human_ans.keys():
        v = human_ans[key]
        if '#' in v:
            missed_key.append(key)
        else:
            human_point.append(int(v))
    for key in cur_ans.keys():
        v = cur_ans[key]
        if key not in missed_key:
            cur_point.append(v)

    distance = math.sqrt(sum([(x - y) ** 2 for x, y in zip(human_point, cur_point)]))

    print('Dis: ', distance)


    cur_dict = {'culture': culture, 'engine': engine, 'distance': distance}

    with open(output_path, mode='a', encoding='utf-8') as f:
        json.dump(cur_dict, f)
        f.write('\n')

    with open(output_path, 'r', encoding='utf-8') as file:
        distances = []
        for line in file:
            data = json.loads(line)
            distances.append(float(data['distance']))

    print('Average Distance: ', sum(distances) / len(distances))

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--output_path', type=str, required=True)
    args = paser.parse_args()
    for culture in country_dict.keys():
        run(culture, output_path=args.output_path, engine="qwen2.5")


    with open(args.ouput_path, 'r', encoding='utf-8') as file:
        distances = []
        for line in file:
            data = json.loads(line)
            distances.append(float(data['distance']))

    print('Average Distance: ', sum(distances) / len(distances))