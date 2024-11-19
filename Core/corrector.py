from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import torch
import argparse
import numpy as np

import jsonlines

from Scorer.factuality_scorer import get_factuality_score
from Scorer.cons_scorer import get_ctrl_score
from Scorer.similarity_score import Sent_Similar

class Prompter:
    def __init__(self, default_prompt=''):
        self.default_prompt = default_prompt

    def generate_prompt(self, instruction, question):
        return f"{instruction} {question}"

def encode_prompt(tokenizer, prompt, max_length):
    return tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

def generate_response(model, tokenizer, prompt, max_new_tokens = 512, top_k = 50, top_p = 0.95):
    inputs = encode_prompt(tokenizer, prompt, max_new_tokens//2)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # max_length=max_length,
            max_new_tokens = max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    return  generated_text[prompt_length + 1:]


def loop_corrector(args, question, bg_prompt, bg_knowledge, model, tokenizer):
    final_answer = ""
    fact_score_loop_num = 0
    consist_loop_num = 0
    entail_loop_num = 0
    fact_score = -100.0
    consist_score = -100.0
    entail_score= -100.0

    bg_knowledge_list = []
    knowledge_score_dict = {} # backgroud_knowledge index->fact score
    best_bg_knowledge = ""

    answer_prompt = f"Refer to the background knowledge: {best_bg_knowledge}, please answer the question with one paragraph: {question}"
    final_answer = generate_response(model, tokenizer, answer_prompt)
    answer_score_dict = {} # final answer index -> consistency score
    final_answer_list = []
    best_final_answer = ""

    # 1.factuality scorer
    while fact_score_loop_num < args.max_loop and fact_score < args.threshold_fact:
        fact_score = get_factuality_score(args.gptscore_model_name, bg_prompt, bg_knowledge)
        bg_knowledge_list.append(bg_knowledge)
        knowledge_score_dict[fact_score_loop_num] = fact_score
        if fact_score < args.threshold_fact:
            refine_prompt = f"The facutuality score for this knowlege: {bg_knowledge} is two low. Please refine the knowlege to improve its factuality"
            bg_knowledge = generate_response(model, tokenizer, refine_prompt)
        # print(f"current loop index: {fact_score_loop_num} fact_score: {fact_score}")
        fact_score_loop_num += 1

    # Get the background knowledge in the highest fact score
    sorted_knowledge_score = sorted(knowledge_score_dict.items(), key=lambda item: item[1], reverse=True)
    best_index = int(sorted_knowledge_score[0][0])
    best_bg_knowledge = bg_knowledge_list[best_index]
    # print(f"best index in fact score: {best_index}")

    # # 2.consistency scorer
    # while consist_loop_num < args.max_loop and consist_score < args.threshold_consistency:
    #     consist_score = get_ctrl_score(answer_prompt, final_answer)
    #     if np.isnan(consist_score):
    #         consist_score = 0
    #     final_answer_list.append(final_answer)
    #     answer_score_dict[consist_loop_num] = consist_score
    #     if consist_score < args.threshold_consistency:
    #         refine_prompt = f"The consistency score between the knowlege: {best_bg_knowledge} and answer: {final_answer} is two low. Please refine the answer to improve its consistency"
    #         final_answer = generate_response(model, tokenizer, refine_prompt, max_new_tokens=640)
    #     print(f"current loop index: {consist_loop_num} consist_score: {consist_score}")
    #     consist_loop_num += 1
    # sorted_final_answer_score = sorted(answer_score_dict.items(), key=lambda item: item[1], reverse=True)
    # best_index = int(sorted_final_answer_score[0][0])
    # best_final_answer = final_answer_list[best_index] 
    # # print(f"best index in consist score: {best_index}")

    # 3.entailment scorer(if you use this scorer, please ban consistency scorer.)
    while entail_loop_num < args.max_loop and entail_score < args.threshold_entailment:
        entail_scores, _ = Sent_Similar().get_scores(answer_prompt, [final_answer])
        entail_score = entail_scores[0]
        if np.isnan(entail_score):
            entail_score = 0
        final_answer_list.append(final_answer)
        answer_score_dict[entail_loop_num] = entail_score
        if entail_score < args.threshold_entailment:
            refine_prompt = f"The entailment score between the knowlege: {best_bg_knowledge} and answer: {final_answer} is two low. Please refine the answer to improve its entailment"
            final_answer = generate_response(model, tokenizer, refine_prompt, max_new_tokens=640)
        # print(f"current loop index: {entail_loop_num} entail_score: {entail_score}")
        entail_loop_num += 1
    sorted_final_answer_score = sorted(answer_score_dict.items(), key=lambda item: item[1], reverse=True)
    best_index = int(sorted_final_answer_score[0][0])
    # print(f"best_final_answer_index: {best_index}")
    best_final_answer = final_answer_list[best_index] 

    return best_final_answer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--threshold_fact", type=float, default=-1)
    parser.add_argument("--threshold_consistency", type=float, default=-5)
    parser.add_argument("--threshold_entailment", type=float, default=0.8)
    parser.add_argument("--max_loop", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="Locutusque/gpt2-large-medical") # prompt model
    parser.add_argument("--gptscore_model_name", type=str, default="opt-350m")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    with jsonlines.open(args.input_file) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            question = line['question']
            prompter = Prompter('')
            instruction = "Provide background knowledge to answer the following question."
            prompt = prompter.generate_prompt(instruction, question)
            bg_knowledge = generate_response(model, tokenizer, prompt, max_new_tokens=256)
            
            response = loop_corrector(args, question, prompt, bg_knowledge, model, tokenizer)

            line.update({'generated_answer': response})
            writer = jsonlines.open(args.output_file, mode='a')
            writer.write(line)
            writer.close()


    
  