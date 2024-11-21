# Please do not run this baseline twice if there is alreay a same name output file!

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import torch
import argparse
import os

import jsonlines

class Prompter:
    def __init__(self, default_prompt=''):
        self.default_prompt = default_prompt

    def generate_prompt(self, instruction, question):
        return f"{instruction} {question}"

def encode_prompt(tokenizer, prompt):
    return tokenizer.encode_plus(prompt, return_tensors='pt')

def generate_response(model, tokenizer, prompt, max_new_tokens = 256, top_k = 50, top_p = 0.95):
    inputs = encode_prompt(tokenizer, prompt)
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

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) # project relative path
    parser.add_argument("--output_file", type=str) # project relative path
    parser.add_argument("--model_name", type=str, default="Locutusque/gpt2-large-medical")
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
            instruction = "Answer the following question."
            prompt = prompter.generate_prompt(instruction, question)
            response = generate_response(model, tokenizer, prompt)

            line.update({'generated_answer': response})
            writer = jsonlines.open(args.output_file, mode='a')
            writer.write(line)
            writer.close()

    
  