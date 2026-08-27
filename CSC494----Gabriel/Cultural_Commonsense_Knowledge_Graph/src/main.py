import json
from datetime import datetime
from utils.llm_query import *
from utils.response_parser import *
import pandas as pd
import numpy as np
import os 
from openpyxl import load_workbook
from config import locations, location_in_local_language,location_to_language,sub_topics_in_local_language_per_location,chinese_monolingual_sutopics,japanese_monolingual_sutopics
from utils.prompt_templates import generation_prompts, extension_prompts
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def log_raw_execution(args, location, sub_topic, raw_response, parsed_data):
    """Logs the exact outputs and hallucinations for methodology documentation."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": args.action,
        "location": location,
        "sub_topic": sub_topic,
        "raw_response": raw_response,
        "success": bool(parsed_data),
        "parsed_item_count": len(parsed_data) if parsed_data else 0
    }
    
    log_filename = f"{args.record_file_name}_raw_logs_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def prepare_prompt(prompt_type, **kwargs):
    if prompt_type in extension_prompts:
        template=extension_prompts[prompt_type]
    elif prompt_type in generation_prompts:
        template=generation_prompts[prompt_type]
    else:
        raise ValueError("Invalid prompt type")
    formatted_prompts = [
        {"role": message["role"], "content": message["content"].format(**kwargs)}
        for message in template ]
    return formatted_prompts

def generate_cultural_commonsense(args,location, sub_topic, premise=None, action=None,knowledge=None,if_then_event=None,tokenizer=None, model=None, llm=None):
    if args.mode=='monolingual_setting':
        if args.action=='initial_generation':
            key_word= 'England_gen'
            prompt=prepare_prompt(key_word,location=location,premise=premise,sub_topic=sub_topic,language='English')
        elif args.action=='relation_extension':
            key_word= 'England_ext'
            prompt=prepare_prompt(key_word,location=location, sub_topic=sub_topic,
                              initial_event=if_then_event, init_action=action, init_knowledge=knowledge,language='English')
    elif args.mode=='multilingual_setting':
        country=location_in_local_language[location]
        language=location_to_language[location]
        if args.action=='initial_generation':
            key_word= location+'_'+'gen'
            prompt=prepare_prompt(key_word,location=country,sub_topic=sub_topic,language=language)
        elif args.action=='relation_extension':
            key_word= location+'_'+'ext'
            prompt=prepare_prompt(key_word,location=country,sub_topic=sub_topic,
                              initial_event=if_then_event, init_action=action, init_knowledge=knowledge,language=language)
    else:
        raise ValueError(f"Unknown mode: {args.mode} please take a look of your mode")
    
    if not args.model.startswith('gpt'):
        response_text=query_llama(prompt=prompt,model_name=args.model,temp=args.temp, model=model, tokenizer=tokenizer,llm=llm)
    else:
        response_text=query_gpt(prompt=prompt, engine=args.model,temp=args.temp)

    try:
        structure_data = parse_llm_response(args, response_text)
    except Exception as e:
        print(f"Parsing failed for {location} ({sub_topic}). Error: {e}")
        structure_data = []
    
    # Log the raw LLM output here for methodology error analysis
    log_raw_execution(args, location, sub_topic, response_text, structure_data)
    
    if args.action == 'relation_extension' and isinstance(structure_data, tuple) and len(structure_data) == 2:
        intermediate_steps, next_steps = structure_data
        merge = intermediate_steps + next_steps
    elif args.action == 'initial_generation' and isinstance(structure_data, list) and structure_data:
        merge = structure_data
    else:
        merge = []

    for entry in merge:
        entry['location']=location
        entry['sub_topic']=sub_topic
        
    #just for test
    if len(structure_data)<2:
        print('len-structure',len(structure_data))
    return structure_data

def save_to_excel(data_sheets, file_name,extend=False):
    columns = ['event', 'knowledge', 'relation', 'llm_result', 'location', 'sub_topic']
    if os.path.exists(file_name):
        existing_workbook=load_workbook(file_name)
        if extend==True:
            if_sheet_exist='replace'
        else:
            if_sheet_exist='overlay'
        with pd.ExcelWriter(file_name,engine='openpyxl',mode='a',if_sheet_exists=if_sheet_exist) as writer:
            writer._book=existing_workbook
            writer._sheets ={ws.title:ws for ws in existing_workbook.worksheets}
        
            for sheet_name,data in data_sheets.items():
                if sheet_name in writer._sheets:
                    print(f"udpating sheet:{sheet_name}")
                else:
                    print(f"adding new sheet : {sheet_name}")
                df=pd.DataFrame(data,columns=columns)
                df.to_excel(writer,sheet_name=sheet_name,index=False)
    else:
        with pd.ExcelWriter(file_name,engine='openpyxl') as writer:
            for sheet_name, data in data_sheets.items():
                df=pd.DataFrame(data,columns=columns)
                df.to_excel(writer,sheet_name=sheet_name,index=False)

def filter_knowledge(data):
    """
    Filters knowledge entries ensuring that all necessary keys are present and non-empty.
    """
    required_keys = ['event', 'knowledge', 'relation', 'llm_result','location', 'sub_topic']
    filtered_data = [item for item in data if all(item.get(key) for key in required_keys)]
    return filtered_data

def process_first_expand_iteration(data,args,tokenizer=None, model=None,llm=None):
    all_row_intermediate_knowledges=[]
    all_row_new_knowledges=[]
    for row in data:
        action=row[0]
        relation=row[2]
        if_then_event=row[3]
        loc=row[4]
        subTopic=row[5]
        # First extension : we extend just relation xNext and oNext
        if relation=='xNext' or relation=='oNext':
            knowledge=row[1]
            result=generate_cultural_commonsense(args,location=loc,sub_topic=subTopic,
                                                                                     action=action,knowledge=knowledge, 
                                                                                     if_then_event=if_then_event, tokenizer=tokenizer, model=model,llm=llm)
            if result and len(result) == 2:
                intermediate_commonsenses, next_commonsenses = result
                if intermediate_commonsenses:
                    filtered_intermediate=filter_knowledge(intermediate_commonsenses)
                    if filtered_intermediate:
                        for item in filtered_intermediate:
                            all_row_intermediate_knowledges.append([item['event'], item['knowledge'], item['relation'], 
                                                                item['llm_result'], item['location'], item['sub_topic']])
                if next_commonsenses: 
                    filtered_commonsense =filter_knowledge(next_commonsenses)
                    if filtered_commonsense:
                        row_new_knowledges=np.array([[item['event'],item['knowledge'],item['relation'],item['llm_result'],
                                                        item['location'],item['sub_topic']]for item in filtered_commonsense])
                        all_row_new_knowledges.append(row_new_knowledges)
                    else:
                        print("Filtered commonsense data has missing or empty values, skipping this entry.")
                else:
                    print("Commonsenses is empty, skipping this entry.")
            else:
                print("Result from generate_cultural_commonsense is invalid or empty, skipping this entry.")
    
    return all_row_intermediate_knowledges,all_row_new_knowledges
    
def expand_knowledge(list_of_knowledge_to_expand,args,tokenizer=None, model=None,llm=None):
    all_row_new_knowledges=[]
    all_row_intermediate_knowledges=[]
    for kg_set in list_of_knowledge_to_expand:
        for row in kg_set:
            action=row[0]
            knowledge=row[1]
            if_then_event=row[3]
            loc=row[4]
            subTopic=row[5]
            print('-----++++++++-sub-topic: {}-++++++++++---'.format(subTopic))
                
            result=generate_cultural_commonsense(args,location=loc,sub_topic=subTopic,
                                                                                     action=action, knowledge=knowledge, 
                                                                                     if_then_event=if_then_event,tokenizer=tokenizer, model=model,llm=llm)
            if result and len(result) == 2:  
                intermediate_commonsenses, next_commonsenses = result

                if intermediate_commonsenses:
                    filtered_intermediate=filter_knowledge(intermediate_commonsenses)
                    if filtered_intermediate:
                        for item in filtered_intermediate:
                            all_row_intermediate_knowledges.append([item['event'], item['knowledge'], item['relation'], 
                                                                item['llm_result'], item['location'], item['sub_topic']])
                if next_commonsenses: 
                    filtered_commonsense = filter_knowledge(next_commonsenses)
                    if filtered_commonsense:
                        row_new_knowledges=np.array([[item['event'],item['knowledge'],item['relation'],item['llm_result'],
                                                        item['location'],item['sub_topic']]for item in filtered_commonsense])
                        all_row_new_knowledges.append(row_new_knowledges)
                    else:
                        print("Filtered commonsense data has missing or empty values, skipping this entry.")
                else:
                    print("Commonsenses is empty, skipping this entry.")
            else:
                print("Result from generate_cultural_commonsense is invalid or empty, skipping this entry.")
    
    return all_row_intermediate_knowledges,all_row_new_knowledges


def extend_relation(model,initial_commonsense_data,args,tokenizer=None, model1=None, llm=None):
    data=initial_commonsense_data.values
    iter_all_new_knowledges=[]
    steps=args.number_extension
    data_sheets={}
    threshold=0.8
    for iter in range(steps):
         # First extension : we extend just relation xNext and oNext
        if iter==0:
            intermediate_all_new_knowledges, iter_all_new_knowledges=process_first_expand_iteration(data,args, tokenizer=tokenizer,
                                                                                                    model=model1, llm=llm)
            #update intermediate knowledge in the dataset
            new_rows=np.array(intermediate_all_new_knowledges)
            if new_rows.size > 0:
                data=np.vstack((data,new_rows))
        else:
            # Filtering to see if the new knowledge is already present or not in the previous iteration and select new knowleges to expand
            pairs_to_add=[]
            list_of_knowledge_to_expand=[]
            events=data[:, 0]
            knowledges=data[:, 1]
            union_unique_values = np.unique(np.concatenate((events, knowledges)))
            union_unique_values_embed=model.encode(union_unique_values)

            for row_knowledges in iter_all_new_knowledges:
                knowledge_to_expand=[]
                row_new_knowledges_embedding=model.encode(row_knowledges[:,1])
                similarities_btw_union_new_knwolge = model.similarity(row_new_knowledges_embedding, union_unique_values_embed)
                for k_idx,knowledge in enumerate(row_knowledges[:,1]):
                    matched=False
                    for e_idx, union in enumerate(union_unique_values):
                        similarity=similarities_btw_union_new_knwolge[k_idx][e_idx]
                        if similarity > threshold :
                            pairs_to_add.append([row_knowledges[k_idx,0], union_unique_values[e_idx], 
                                                  row_knowledges[k_idx,2],row_knowledges[k_idx,3],
                                                  row_knowledges[k_idx,4],row_knowledges[k_idx,5]])
                            matched=True
                            break
                    if not matched:
                        pairs_to_add.append(row_knowledges[k_idx])
                        if k_idx<=5:
                            print('{} element added'.format(k_idx))
                            knowledge_to_expand.append(row_knowledges[k_idx])

                list_of_knowledge_to_expand.append(np.array(knowledge_to_expand))
            # update data with new knowledges
            new_rows=np.array(pairs_to_add)
            if new_rows.size > 0:
                data=np.vstack((data,new_rows))
            # temporary save
            location=data[0, 4]
            data_sheets[location]=data
            record_file_name= args.record_file_name +'_temp'
            save_to_excel(data_sheets,record_file_name+'.xlsx',extend=True)
            # expand the new Knowlege 
            if iter < steps-1:
                intermediate_all_new_knowledges,iter_all_new_knowledges=expand_knowledge(list_of_knowledge_to_expand,args,tokenizer=tokenizer, model=model1, llm=llm)
                #update intermediate knowledge in the dataset
                new_rows=np.array(intermediate_all_new_knowledges)
                if new_rows.size > 0:
                    data=np.vstack((data,new_rows))
    return data


def add_params():
    parser= argparse.ArgumentParser()
    parser.add_argument("--record_file_name",type=str, help='name on which you want to save your file')
    parser.add_argument("--initial_data_path",type=str,help='path of the intial commonsense data file')
    parser.add_argument('--temp', type=float, default=1, help='temperature for generation (higher=more diverse)')
    parser.add_argument("--number_location",type=int, default=None, help="number of location to process in the extension of relation")
    parser.add_argument("--number_extension",type=int, help="number of time to extend the relation",default=3)
    parser.add_argument("--number_subtopic",type=int, default=None,help="number of topic to process")
    parser.add_argument("--model",type=str,default='gpt-4o',help='model we want to use to generate ckg')
    parser.add_argument("--sub_sample",action='store_true',help='run a sub sample of data in the extension phase')
    parser.add_argument("--mode",type=str,default='monolingual',help='run monolingual(english for all location) or multilingual(each location with his local language)',
                        choices=['monolingual_setting', 'multilingual_setting'])
    parser.add_argument("--action",type=str,help='choose action to perform between intiatial extraction or relation extensions',
                        choices=['initial_generation', 'relation_extension'],default='initial_generation')
    
    params=parser.parse_args()
    return params


if __name__ =='__main__':

    args=add_params()
    tokenizer=None 
    model=None 
    llm=None

    # Modify next line is you want to add more models for the generation 
    if "meta-llama" in args.model  or "google" in args.model  :
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(model=args.model)
    elif not args.model.startswith('gpt'):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model=AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

    if args.number_location is None:
        args.number_location=len(locations)
        
    # Extract the initial if then cultural commonsense knowledges
    if args.action=='initial_generation':
        for index in range(args.number_location):
            data_sheets={}
            location=locations[index]
            print('location', location)
            if args.mode=='multilingual_setting':
                if location=='England':
                    continue
                sub_topics=sub_topics_in_local_language_per_location[location]
            else:
                if location=='China':
                    sub_topics=chinese_monolingual_sutopics
                elif location=='Japan':
                    sub_topics=japanese_monolingual_sutopics
                else:
                    sub_topics=sub_topics_in_local_language_per_location['England']    
            if args.number_subtopic is not None:
                sub_topics=sub_topics[:args.number_subtopic]    
            all_data = []
            for subTopic in sub_topics:
                print(subTopic)
                for _ in range(0,1):
                    result=generate_cultural_commonsense(args,location,subTopic,  tokenizer=tokenizer, model=model, llm=llm)
                    if result:
                        commonsense=result
                        all_data.extend(commonsense)
                    else:
                        print("Result from generate_cultural_commonsense is invalid or empty, skipping this entry.")
            data_sheets[location]=all_data
            record_file_name= args.record_file_name
            save_to_excel(data_sheets,record_file_name+'.xlsx',extend=True)

    elif args.action=='relation_extension':
        # Extend the xNext/oNext relations
        if args.mode=='monolingual_setting':
            model=SentenceTransformer("all-MiniLM-L6-v2")
        elif args.mode=='multilingual_setting':
            model=SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")
        for index in range(args.number_location):
            data_sheets={}
            location=locations[index]
            print('location', location)
            if args.mode=='multilingual_setting':
                if location=='England':
                    continue
            initial_commonsense_data=pd.read_excel(args.initial_data_path,sheet_name=location)
            if args.sub_sample:
                x_next_df = initial_commonsense_data[(initial_commonsense_data['relation'] == 'xNext') | (initial_commonsense_data['relation'] == 'oNext')]
                initial_commonsense_data=sample_subtopics(x_next_df.head(55),10)
            expanded_data=extend_relation(model,initial_commonsense_data,args, tokenizer=tokenizer, model1=model, llm=llm)
            data_sheets[location]=expanded_data
            record_file_name= args.record_file_name
            save_to_excel(data_sheets,record_file_name+'.xlsx',extend=True)
    
    print('done')