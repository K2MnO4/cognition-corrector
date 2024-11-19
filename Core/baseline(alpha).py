from tqdm import tqdm
import torch
import jsonlines
import argparse

from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

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


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str) # project relative path
    parser.add_argument("--output_file", type=str) # project relative path
    parser.add_argument("--model_name", type=str, default="baffo32/decapoda-research-llama-7B-hf")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            instruction = "Answer the following question."
            prompt = prompter.generate_prompt(instruction, question)
            # print(f"prompt is***:{prompt}")
            response = generate_response(model, tokenizer, prompt)
            # print(f"*response is* {response}")

            line.update({'generated_answer': response})
            writer = jsonlines.open(args.output_file, mode='a')
            writer.write(line)
            writer.close()


        
            
                