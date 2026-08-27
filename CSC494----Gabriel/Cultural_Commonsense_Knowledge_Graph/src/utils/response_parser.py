import json
import argparse
import pandas as pd
import re

def extract_json_block(text):
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index]
    else:
        raise ValueError("No valid JSON block found.")


def clean_output(response_text):
    start_index=response_text.find('[')
    if start_index !=-1:
        end_index=response_text.rfind(']',start_index)+1
        if end_index != -1:
            clean_json_string = response_text[start_index:end_index]
            try:
                return clean_json_string
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format: {e}")
                return None
        else:
            print("No closing bracket found for JSON.")
            return None
    else:
        print("No valid JSON found.")
        return None

def extract_json_objects(text):
    """
    Extract JSON-like objects from a malformed text response.
    """
    #regex for extracting Json object
    object_pattern = re.compile(r'\{[^{}]*\}') 
    extracted_objects = object_pattern.findall(text)
    # remove space and \n before keys
    cleaned_json_list = [s.replace("\n", "").replace("    ", "").strip() for s in extracted_objects] 
    return cleaned_json_list

def fix_json_object(args, bad_json_obj):
    #Remove all quotes
    text = re.sub(r'["\']', '', bad_json_obj)
    result = {}
    #  threat first result or event field
    if args.action == 'initial_generation':
        result_pattern = r'\bresult\s*:\s*(If[^}]+)'
    elif args.action == 'relation_extension':
        result_pattern = r'\bevent\s*:\s*(If[^}]+)'
    result_match = re.search(result_pattern, text)
    if result_match:
        if args.action == 'initial_generation':
            result['result'] = result_match.group(1).strip()
        elif args.action == 'relation_extension':
            result['event'] = result_match.group(1).strip()
        #extract result or event field from text to avoid conflict
        text = re.sub(result_pattern, '', text)
    # extract the others part
    pattern = r'\b(action|knowledge|relation_type)\b\s*:\s*([^,}\n]+)'
    matches = re.finditer(pattern, text)
    for match in matches:
        key = match.group(1)
        value = match.group(2).strip()
        result[key] = value
    return json.dumps(result)

def extract_valid_json_objects(args,response_text):
    if args.action  == 'initial_generation':
        required_keys = {"action", "knowledge", "relation_type", "result"}
    elif args.action  == 'relation_extension':
        required_keys = {"action", "knowledge", "relation_type", "event"}
        
    extracted_objects = extract_json_objects(response_text)
    valid_objects = []
    for obj in extracted_objects:
        fixed_json = fix_json_object(args,obj)
        #print(fix_json_object)
        try:
            json_obj = json.loads(fixed_json)
            if required_keys.issubset(json_obj.keys()):
                valid_objects.append(json_obj)
        except json.JSONDecodeError:
            print('error')
            #Ignore Json with invalid field
            continue 
    
    return valid_objects

def parse_llm_response(args, response_text):
    commonsense_data = []
    intermediaire_events = []
    next_events = []
    try:
        clean_response_text = clean_output(response_text)
        commonsenses = json.loads(clean_response_text)
    except json.JSONDecodeError:
        if args.action == 'relation_extension':
            json_content = extract_json_block(response_text)
            # extraction of keys and values
            # pattern = re.compile(r'(?P<key>intermediate_steps|next_steps)\s*:\s*(?P<values>\[[^\]]*\])')
            # if encounter error uncomment the next line and comment the line above
            pattern = re.compile(r'"?(?P<key>intermediate_steps|next_steps)"?\s*:\s*(?P<values>\[[^\]]*\]|\[\s*(?:\{.*?\}\s*,?\s*)+\])', re.DOTALL)
            matches = pattern.findall(json_content)
            print(matches)
            dic={}
            for key, values in matches:
                print(key)
                dic[key]=extract_valid_json_objects(args,values)
            commonsenses=  [dic]
        elif args.action == 'initial_generation':
            commonsenses = extract_valid_json_objects(args,response_text)

    print('====== COMOMESENSE====', commonsenses)
    if args.action == 'initial_generation':
        for commonsense in commonsenses:
            if isinstance(commonsense, dict):
                event = commonsense.get("action", "")
                knowledge = commonsense.get("knowledge", "")
                relation_type = commonsense.get("relation_type", "")
                if_then_event = commonsense.get("result", "")

                if not all([event, knowledge, relation_type, if_then_event]):
                    continue

                commonsense_data.append({
                    "event": event.strip(),
                    "knowledge": knowledge.strip(),
                    "relation": relation_type.strip(),
                    "llm_result": if_then_event.strip()
                })
        return commonsense_data

    elif args.action == 'relation_extension':
        for key in commonsenses[0].keys():
            items_list = commonsenses[0].get(key, [])
            if not isinstance(items_list, list):
                continue
            for items in items_list:
                event = items.get("action", "")
                knowledge = items.get("knowledge", "")
                relation_type = items.get("relation_type", "")
                if_then_event = items.get("event", "")

                if not all([event, knowledge, relation_type, if_then_event]):
                    continue

                if key == 'intermediate_steps':
                    intermediaire_events.append({
                        "event": event.strip(),
                        "knowledge": knowledge.strip(),
                        "relation": relation_type.strip(),
                        "llm_result": if_then_event.strip()
                    })
                elif key == 'next_steps':
                    next_events.append({
                        "event": event.strip(),
                        "knowledge": knowledge.strip(),
                        "relation": relation_type.strip(),
                        "llm_result": if_then_event.strip()
                    })
        return intermediaire_events, next_events

def sample_subtopics(dataframe, total_samples, subtopic_column='sub_topic', random_state=42):
    """
    Samples rows from a DataFrame based on subtopics, ensuring a minimum per subtopic 
    and filling up to the desired total number of samples.

    Args:
    - dataframe (pd.DataFrame): The input DataFrame.
    - total_samples (int): Desired total number of samples.
    - subtopic_column (str): Column indicating subtopics.
    - random_state (int, optional): Seed for reproducibility. Default is 42.

    Returns:
    - pd.DataFrame: The sampled DataFrame.
    """
    min_per_subtopic = max(1, total_samples // dataframe[subtopic_column].nunique())
    grouped = dataframe.groupby(subtopic_column, group_keys=False)
    samples = grouped.apply(lambda x: x.sample(n=min(len(x), min_per_subtopic), random_state=random_state))
    remaining = total_samples - len(samples)
    if remaining > 0:
        additional_samples = dataframe.drop(samples.index).sample(n=remaining, random_state=random_state)
        samples = pd.concat([samples, additional_samples])
    return samples




