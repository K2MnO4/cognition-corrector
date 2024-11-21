from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

import torch
import argparse
import numpy as np

import jsonlines

from Scorer.factuality_scorer import get_factuality_score
from Scorer.cons_scorer import get_ctrl_score
from Scorer.similarity_score import Sent_Similar

from Utils.utils.prompter import Prompter


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        early_stopping=True,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )
    generated_text = generation_output.sequences[0][len(input_ids[0]):]
    output = tokenizer.decode(generated_text)
    return output


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
    prompter = Prompter('')


    # 1.factuality scorer
    while fact_score_loop_num < args.max_loop and fact_score < args.threshold_fact:
        fact_score = get_factuality_score(args.gptscore_model_name, bg_prompt, bg_knowledge)
        # print(f"current loop index: {fact_score_loop_num} \n fact_score: {fact_score} \n knowlege: {bg_knowledge}")
        bg_knowledge_list.append(bg_knowledge)
        knowledge_score_dict[fact_score_loop_num] = fact_score
        if fact_score < args.threshold_fact:
            # refine_prompt = f"The facutuality score for knowlege: {bg_knowledge} is two low. Please refine the knowlege to improve its factuality"
            refine_prompt = prompter.generate_prompt("Please refine the following backgroud knowlege to improve its factuality", f"{bg_knowledge}")
            bg_knowledge = generate_response(model, tokenizer, refine_prompt)
        fact_score_loop_num += 1

    # Get the background knowledge in the highest fact score
    sorted_knowledge_score = sorted(knowledge_score_dict.items(), key=lambda item: item[1], reverse=True)
    best_index = int(sorted_knowledge_score[0][0])
    best_bg_knowledge = bg_knowledge_list[best_index]
    # print(f"best_bg_knowlege: {best_bg_knowledge}")
    print(f"best index in fact score: {best_index}")

    # parameters for 2 and 3 scorer
    answer_prompt = prompter.generate_prompt("Please answer the question with one paragraph", f"Refer to the background knowledge: {best_bg_knowledge}, {question}")
    final_answer = generate_response(model, tokenizer, prompt)
    answer_score_dict = {} # final answer index -> consistency score
    final_answer_list = []
    best_final_answer = ""


    # 2.consistency scorer
    while consist_loop_num < args.max_loop and consist_score < args.threshold_consistency:
        consist_score = get_ctrl_score(best_bg_knowledge, final_answer)
        # print(f"current loop index: {consist_loop_num} \n consist_score: {consist_score} \n answer: {final_answer}")
        if np.isnan(consist_score):
            consist_score = 0
        final_answer_list.append(final_answer)
        answer_score_dict[consist_loop_num] = consist_score
        if consist_score < args.threshold_consistency:
            # refine_prompt = f"The consistency score between the knowlege: {best_bg_knowledge} and response: {final_answer} is two low. Please refine the response to improve its consistency"
            refine_prompt = prompter.generate_prompt(f"The consistency score between response: {final_answer} and backgroud knowledge: {best_bg_knowledge} is too low, please refine response to answer the following question", question)
            final_answer = generate_response(model, tokenizer, refine_prompt)
        consist_loop_num += 1
    sorted_final_answer_score = sorted(answer_score_dict.items(), key=lambda item: item[1], reverse=True)
    best_index = int(sorted_final_answer_score[0][0])
    best_final_answer = final_answer_list[best_index] 
    print(f"best index in consist score: {best_index}")
    # print(f"***best_ans: {best_final_answer}")

    # 3.entailment scorer(if you use this scorer, please ban consistency scorer.)
    # while entail_loop_num < args.max_loop and entail_score < args.threshold_entailment:
    #     entail_scores, _ = Sent_Similar().get_scores(answer_prompt, [final_answer])
    #     entail_score = entail_scores[0]
    #     if np.isnan(entail_score):
    #         entail_score = 0
    #     final_answer_list.append(final_answer)
    #     answer_score_dict[entail_loop_num] = entail_score
    #     if entail_score < args.threshold_entailment:
    #         # refine_prompt = f"The entailment score between the knowlege: {best_bg_knowledge} and answer: {final_answer} is two low. Please refine the answer to improve its entailment"
    #         refine_prompt = prompter.generate_prompt("Please refine the answer to improve its entailment", f"The entailment score between the knowlege: {best_bg_knowledge} and answer: {final_answer} is two low.")
    #         final_answer = generate_response(model, tokenizer, refine_prompt)
    #     print(f"current loop index: {entail_loop_num} entail_score: {entail_score}")
    #     entail_loop_num += 1
    # sorted_final_answer_score = sorted(answer_score_dict.items(), key=lambda item: item[1], reverse=True)
    # best_index = int(sorted_final_answer_score[0][0])
    # print(f"best_final_answer_index: {best_index}")
    # best_final_answer = final_answer_list[best_index] 
    # print(f"***best_ans: {best_final_answer}")

    return best_final_answer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--threshold_fact", type=float, default=-1)
    parser.add_argument("--threshold_consistency", type=float, default=-5)
    parser.add_argument("--threshold_entailment", type=float, default=0.8)
    parser.add_argument("--max_loop", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="baffo32/decapoda-research-llama-7B-hf") # prompt model
    parser.add_argument("--gptscore_model_name", type=str, default="opt-350m")
    args = parser.parse_args()

    base_model = args.model_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_8bit = True
    lora_weights = 'tloen/alpaca-lora-7b'

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if load_8bit:
        model.half()
    model.eval()

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


    
  