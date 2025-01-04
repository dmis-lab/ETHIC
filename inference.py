import os
import json
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI

from utils import get_logger, get_model_prompts, calculate_score, create_batch_for_summarizing, run_batch_for_summarizing, parse_score_for_summarizing
from api_config import CONFIG

def main(args):

    # set logger
    model_name = os.path.basename(args.model_name_or_path)

    path_to_logdir = os.path.join(args.log_path, model_name, args.task)
    os.makedirs(path_to_logdir, exist_ok=True)
    logger = get_logger(logger_name=__name__, path_to_logdir=path_to_logdir)

    dataset = load_dataset("dmis-lab/ETHIC", args.task, cache_dir=args.cache_dir)["test"]

    logger.info(f"Loaded dataset for task {args.task}")

    save_path = os.path.join(args.save_path, model_name, args.task)
    os.makedirs(save_path, exist_ok=True)

    os.makedirs(os.path.join(save_path, "Books"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "Debates"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "Medicine"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "Law"), exist_ok=True)

    prompt = get_model_prompts(args.model_name_or_path)

    # load model
    if "gpt" in args.model_name_or_path:
        client = OpenAI(api_key=CONFIG["openai"][0])
    elif "gemini" in args.model_name_or_path:
        genai.configure(api_key=CONFIG["google"][0])
        model = genai.GenerativeModel(args.model_name_or_path)
    else: # vllm
        model = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=4096)
    
    logger.info("Loaded model")

    scores = []
    if args.task == "Organizing":
        score_per_domain = {
            "Books":[],
            "Debates":[],
        }
    elif args.task == "Attributing":
        score_per_domain = {
            "Medicine":[],
            "Law":[]
        }
    else:
        score_per_domain = {
            "Books":[],
            "Debates":[],
            "Medicine":[],
            "Law":[]
        }

    logger.info(f"Saving model predictions to {save_path}")
    
    dataset_tqdm = tqdm(dataset, file=open(os.devnull, "w"))
    for sample in dataset_tqdm:
        
        id_ = sample["ID"]
        answer = sample["Answer"]
        system_msg = sample["System_msg"]
        user_msg = sample["User_msg"]
        domain = sample["Domain"]
        
        logger.info(f"{str(dataset_tqdm)} Domain: {domain}, ID: {id_}")

        if "gemini" in args.model_name_or_path:
            full_prompt = prompt.format(system_msg=system_msg, user_msg=user_msg)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=4096,
                    temperature=0.0
                ),
                safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            )
            try:
                prediction = response.text
            except ValueError: # gemini models occasionally refuse to answer 
                logger.warning("Prediction FAILED")
                prediction = "FAILED"
        elif "gpt" in args.model_name_or_path:
            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=[
                    {"role" : "system", "content": system_msg},
                    {"role":"user", "content":user_msg}
                ],
                temperature=0,
                top_p=1.0,
                max_tokens=4096,
            )
            prediction = completion.choices[0].message
        elif "Qwen" in args.model_name_or_path or "glm" in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            messages=[
                {"role" : "system", "content": system_msg},
                {"role":"user", "content":user_msg}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = model.generate([full_prompt], sampling_params)
            for output in outputs:
                prediction = output.outputs[0].text
        else:
            full_prompt = prompt.format(system_msg=system_msg, user_msg=user_msg)
            outputs = model.generate([full_prompt], sampling_params)
            for output in outputs:
                prediction = output.outputs[0].text

        result_dict, score = calculate_score(args.task, domain, user_msg, prediction, answer)
        scores.append(score)
        score_per_domain[domain].append(score)

        with open(os.path.join(save_path, domain, f"{id_}.json"), "w") as wf:
            json.dump(result_dict, wf)

    # for Summarizing task, score using batch inference
    
    if args.task == "Summarizing":

        assert scores == [], "Initial total score for Summarizing should be empty!"

        logger.info("Preparing for summary scoring (batch inference)")

        path_list = [str(f) for f in Path(save_path).rglob("*.json") if f.parent.name in ["Books", "Debates", "Law", "Medicine"]]
        batch_for_summarizing = create_batch_for_summarizing(path_list)

        batch_input_path = os.path.join(os.path.dirname(save_path), "batch_inference", "summarizing_input.jsonl")
        if os.path.exists(batch_input_path):
            logger.error(f"Batch file for {model_name} already exists!")
            raise ValueError()
        
        os.makedirs(os.path.dirname(batch_input_path), exist_ok=True)
        
        with open(batch_input_path, "a") as wf:
            for line in batch_for_summarizing:
                wf.write(json.dumps(line) + "\n")

        logger.info("Running batch inference")

        try:
            batch_output_path = run_batch_for_summarizing(batch_input_path)
        except ValueError:
            logger.error("Batch inference FAILED")
            sys.exit(1)

        logger.info("Batch inference COMPLETE")

        score_dict = parse_score_for_summarizing(batch_output_path)
        for domain_filename in score_dict:
            domain = domain_filename[:domain_filename.find("_")]
            filename = domain_filename[domain_filename.find("_")+1:]
            
            filepath = os.path.join(save_path, domain, f"{filename}.json")
            with open(filepath) as rf:
                orig_dict = json.load(rf)
            
            prediction = orig_dict["prediction"]
            input_sections = orig_dict["input_sections"]
            score = score_dict[domain_filename]["weighted"]

            with open(filepath, "w") as wf:
                json.dump({
                    "prediction": prediction,
                    "input_sections": input_sections,
                    "score": score
                }, wf)
            
            scores.append(score)
            score_per_domain[domain].append(score)

    # write score file (overall / per domain)
    avg_score = sum(scores) / len(scores)
    avg_score_per_domain = {key: sum(value) / len(value) for key, value in score_per_domain.items()}

    with open(os.path.join(save_path, "final_score.txt"), "w") as wf:
        wf.write(str(avg_score))
    with open(os.path.join(save_path, "domain_score.json"), "w") as wf:
        json.dump(avg_score_per_domain, wf)
    
    logger.info("All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Choose from [\"Recalling\", \"Summarizing\", \"Organizing\", \"Attributing\"]")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "results"))
    parser.add_argument("--log_path", type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs"))
    
    args = parser.parse_args()
    main(args)