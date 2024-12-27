import re

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

def calculate_score(task, prediction, answer):

    result_dict = dict()
    result_dict["prediction"] = prediction
    result_dict["answer"] = answer

    if prediction == "FAILED": # gemini models refuse to answer at times
        return result_dict, 0
    
    if task == "Recalling":
        result_dict["precision"], result_dict["recall"], result_dict["f1_score"] = calculate_f1_score(prediction, answer)
        score = result_dict["f1_score"]
    elif task == "Summarizing": # will be calculated separately
        score = 0
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