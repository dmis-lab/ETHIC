import re
import json
import os
import time
from openai import OpenAI
import numpy as np
import logging
from datetime import datetime
from api_config import CONFIG

def get_logger(logger_name, path_to_logdir):
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%y%m%d-%H%M")
    path_to_logfile = os.path.join(path_to_logdir, f"{formatted_datetime}.log")

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(
            path_to_logfile,
            mode="a",
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def calculate_f1_score(model_answer, label_list):
    
    model_list = re.split(r"[;,]\s*", model_answer)
    
    model_list = sorted(set([pred.lower().strip(".").strip() for pred in model_list]))
    label_list = sorted([label.lower().strip(".").strip() for label in label_list])

    num_labels = len(label_list)
    tp = 0
    for pred in model_list:
        for label in label_list[:]:
            if pred == label:
                tp += 1
                break
    
    fp = len(model_list) - tp
    fn = num_labels - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # return f1_score
    return precision, recall, f1_score

def calculate_lcs(prediction, answer):

    m = len(prediction)
    n = len(answer)
    
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif prediction[i-1] == answer[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    index = L[m][n]
    
    lcs_sequence = [""] * index
    
    i, j = m, n
    while i > 0 and j > 0:
        if prediction[i-1] == answer[j-1]:
            lcs_sequence[index-1] = prediction[i-1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs_length = len(lcs_sequence)
    lcs_score = lcs_length / len(answer)
    
    return lcs_sequence, lcs_score

def create_batch_for_summarizing(path_list):

    with open("./summeval_prompts/con_detailed.txt") as rf:
        prompt_con = rf.read()
    with open("./summeval_prompts/faith_detailed.txt") as rf:
        prompt_faith = rf.read()
    with open("./summeval_prompts/rel_detailed.txt") as rf:
        prompt_rel = rf.read()
    criteria_dict = {prompt_con: 'con', prompt_faith: 'faith', prompt_rel: 'rel'}
    
    batch_list = []
    for path in path_list:
        with open(path) as rf:
            pred_dict = json.load(rf)
        
        # 1. prepare section-wise context / prediction
        domain = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path).replace(".json", "")

        if domain == "Law":
            section_pattern = re.compile("(<Segment \d+>)")
        else:
            section_pattern = re.compile("(<Section \d+>)")

        section_context_dict = dict()
        section_pred_dict = dict()
        orig_sections = section_pattern.split(pred_dict["input_sections"])
        summ_sections = section_pattern.split(pred_dict["prediction"])
        
        for i in range(1, len(orig_sections), 2):
            section_context_dict[orig_sections[i]] = orig_sections[i+1].strip()
        
        for i in range(1, len(summ_sections), 2):
            section_pred_dict[summ_sections[i]] = summ_sections[i+1].strip()

        # 2. create batch using 3 different criteria per section
        for prompt in [prompt_con, prompt_faith, prompt_rel]:
            for section in section_context_dict:
                if section not in section_pred_dict: # model did not create summary for the section
                    continue
                prompt_with_content = prompt.replace('{{Document}}', section_context_dict[section]).replace('{{Summary}}', section_pred_dict[section])
                batch = {
                    'custom_id': f"{domain}_{filename}_{section}_{criteria_dict[prompt]}",
                    'method': 'POST',
                    'url': "/v1/chat/completions",
                    'body': {
                        'model': 'gpt-4o-2024-08-06',
                        'messages': [{"role": "system", "content": prompt_with_content}],
                        'temperature': 0,
                        'max_tokens': 5,
                        'top_p': 1,
                        'frequency_penalty': 0,
                        'presence_penalty': 0,
                        'stop': None,
                        'logprobs': True,
                        'top_logprobs': 10,
                        'n': 1
                    }
                }

                batch_list.append(batch)
        
    return batch_list
        
def run_batch_for_summarizing(batch_input_path):
    
    batch_output_path = os.path.join(os.path.dirname(batch_input_path), "summarizing_output.jsonl")

    client = OpenAI(api_key=CONFIG["openai"][0])
    batch_input_file = client.files.create(
    file=open(batch_input_path, "rb"),
    purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    time.sleep(10)

    # retrieve batch information
    retrieved_batch_job = client.batches.retrieve(batch_job.id)
    
    while True:
        time.sleep(30) # wait for 30 seconds for another status request
        retrieved_batch_job = client.batches.retrieve(batch_job.id)
        if retrieved_batch_job.status == 'completed' or retrieved_batch_job.status == 'failed':
            break
    
    if retrieved_batch_job.status == 'failed':
        raise ValueError()

    result_file_id = retrieved_batch_job.output_file_id
    result = client.files.content(result_file_id).text

    time.sleep(10)

    with open(batch_output_path, "w") as wf:
        wf.write(result)

    return batch_output_path

def parse_score_for_summarizing(batch_output_path):
    
    batch_outputs = []
    with open(batch_output_path) as rf:
        for line in rf:
            batch_outputs.append(json.loads(line))
    
    samples = dict()
    for batch_output in batch_outputs:
        custom_id = batch_output["custom_id"] # {domain}_{filename}_{section}_{criteria}
        domain = custom_id.split("_")[0]
        section_format_text = "_<Segment" if domain == "Law" else "_<Section"
        filename = custom_id[custom_id.find("_")+1:custom_id.find(section_format_text)]
        section = custom_id[custom_id.find(section_format_text) + 1:custom_id.rfind("_")]
        criteria = custom_id[custom_id.rfind("_")+1:]

        sample_id = f"{domain}_{filename}"
        if sample_id not in samples:
            samples[sample_id] = {'weighted_con': 0,  'weighted_rel': 0, 'weighted_faith': 0, 'top_con': 0, 'top_rel': 0, 'top_faith': 0, 'count': 0}
        
        # Extract scores from the response
        top_logprobs = batch_output['response']['body']['choices'][0]['logprobs']['content'][0]['top_logprobs']
        token_value = batch_output['response']['body']['choices'][0]['logprobs']['content'][0]['token']
        
        # Update top score
        try:
            samples[sample_id][f'top_{criteria}'] += float(token_value)
        except ValueError:
            samples[sample_id][f'top_{criteria}'] += 0
        
        # Calculate weighted scores
        scores_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for tokens in top_logprobs:
            try:
                score = int(tokens['token'])
            except ValueError:
                continue

            if score < 1 or score > 5:
                continue

            logprob = tokens.get('logprob', float('-inf'))
            prob = np.exp(logprob)
            scores_dict[score] += prob

        for score, prob in scores_dict.items():
            samples[sample_id][f'weighted_{criteria}'] += score * prob

        samples[sample_id]['count'] += 1

    # Average scores   
    for sample in samples:
        samples[sample]['count'] /= 3
        
        for score in samples[sample]:
            samples[sample][score] /= samples[sample]['count']
            
        samples[sample]['weighted'] = sum(samples[sample][feature] for feature in samples[sample] if 'weighted' in feature) / 3
        samples[sample]['top'] = sum(samples[sample][feature] for feature in samples[sample] if 'top' in feature) / 3

    return samples

def calculate_score(task, domain, user_msg, prediction, answer):

    result_dict = dict()
    result_dict["prediction"] = prediction
    result_dict["answer"] = answer

    if prediction == "FAILED": # gemini models refuse to answer at times
        return result_dict, 0
    
    if task == "Recalling":
        result_dict["precision"], result_dict["recall"], result_dict["f1_score"] = calculate_f1_score(prediction, answer)
        score = result_dict["f1_score"]
    elif task == "Summarizing": 
        input_sections_or_segments = re.search("### Context:\n(.+?)\n\nNow, respond to the instruction", user_msg, re.DOTALL).group(1)
        result_dict["input_sections"] = input_sections_or_segments
        score = 0 # score will be calculated separately
    elif task == "Organizing":
        pred_in_list = re.findall("\d+", prediction)
        answer_in_list = re.findall("\d+", answer)
        result_dict["lcs"], result_dict["lcs_score"] = calculate_lcs(pred_in_list, answer_in_list)
        score = result_dict["lcs_score"]
    elif task == "Attributing":
        match = re.search(r"(Related Segments|Core IDs):\s*(.+)", prediction)
        if match: # model has followed format instruction
            target_span = match.group(2)
        else:
            target_span = prediction
        
        pred_numbers = ", ".join(set(re.findall("\d+", target_span)))
        answer_numbers = [re.search("\d+", ans).group() for ans in answer]

        ## account for answer=[]
        if pred_numbers == []:
            pred_numbers = "None"
        if answer_numbers == []:
            answer_numbers = ["None"]
        
        result_dict["precision"], result_dict["recall"], result_dict["f1_score"] = calculate_f1_score(pred_numbers, answer_numbers)
        score = result_dict["f1_score"]
    
    return result_dict, score

def get_model_prompts(model_name_or_path):

    if "gemini" in model_name_or_path:
        prompt = "{system_msg}\n\n{user_msg}"
    elif "Llama-3.1" in model_name_or_path:
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Phi" in model_name_or_path:
        prompt = "<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant|>"
    else: # gpt, qwen, glm receives "messages" list as input
        prompt = ""
    return prompt 