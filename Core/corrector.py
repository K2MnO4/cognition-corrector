from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import torch
import argparse

import jsonlines

from Scorer.factuality_scorer import get_factuality_score

class Prompter:
    def __init__(self, default_prompt=''):
        self.default_prompt = default_prompt

    def generate_prompt(self, instruction, question):
        return f"{instruction} {question}"

def encode_prompt(tokenizer, prompt):
    return tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=256)

def generate_response(model, tokenizer, prompt, max_new_tokens = 512, top_k = 50, top_p = 0.95):
    inputs = encode_prompt(tokenizer, prompt)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

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
    gpt_score_loop_num = 0
    answer_loop_num = 0
    entail_loop_num = 0
    fact_score = -100.0
    consist_score = -100.0
    entail_score= -100.0

    bg_knowledge_list = []
    knowledge_score_dict = {} # backgroud_knowledge index->fact score
    best_bg_knowledge = ""


    # 1.factuality scorer
    while gpt_score_loop_num < args.max_loop and fact_score < args.threshold_fact:
        fact_score = get_factuality_score(args.gptscore_model_name, bg_prompt, bg_knowledge)
        bg_knowledge_list.append(bg_knowledge)
        knowledge_score_dict[gpt_score_loop_num] = fact_score
        gpt_score_loop_num += 1
        if fact_score < args.threshold_fact:
            refine_prompt = f"The facutuality score for this knowlege: {bg_knowledge} is so lower. Please refine the knowlege to improve its factuality"
            bg_knowledge = generate_response(model, tokenizer, refine_prompt)

    # Get the background knowledge in the highest fact score
    sorted_knowledge_score_dict = sorted(knowledge_score_dict.items(), key=lambda item: item[1], reverse=True)
    best_index = int(sorted_knowledge_score_dict[0][0])
    best_bg_knowledge = bg_knowledge_list[best_index]

    # Test for generating final answer
    answer_prompt = f"Refer to the background knowledge: {best_bg_knowledge}, please answer the question with one paragraph: {question}"
    final_answer = generate_response(model, tokenizer, answer_prompt)

    # 2.consistency scorer

    # 3.entailment scorer

    return final_answer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--threshold_fact", type=float, default=-1)
    parser.add_argument("--threshold_consistency", type=float, default=-5)
    parser.add_argument("--threshold_entailment", type=float, default=0.8)
    parser.add_argument("--max_loop", type=int, default=1)
    parser.add_argument("--gptscore_model_name", type=str, default="opt-350m")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Locutusque/gpt2-large-medical"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    with jsonlines.open(args.input_file) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            question = line['question']
            prompter = Prompter('')
            instruction = "Provide background knowledge to answer the following question."
            prompt = prompter.generate_prompt(instruction, question)
            bg_knowledge = generate_response(model, tokenizer, prompt)
            
            response = loop_corrector(args, question, prompt, bg_knowledge, model, tokenizer)

            line.update({'generated_answer': response})
            writer = jsonlines.open(args.output_file, mode='a')
            writer.write(line)
            writer.close()


    
  