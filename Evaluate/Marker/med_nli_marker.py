import jsonlines
import numpy as np
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

def classify_text(model, tokenizer, text):
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(model.device)
    attention_masks = encoding["attention_mask"].to(model.device)
    try:
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            max_length=8
        )
    except:
        print("classify text error: ", text, len(tokenizer.encode(text)))
        return None  
    result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if "contradiction" in result:
        return "contradiction", -1
    if "entailment" in result:
        return "entailment", 1
    return "neutral", 0

def truncate(tokenizer, text):
    ids = tokenizer.encode(text)
    return tokenizer.decode(ids[:2000])

def calc_med_nli_score(input_file_path, output_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")   
    model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI").to(device)

    overall_comparison_val_list = []  # judge the relationship between generated answer and golden answer by individual sample
    sentence_comparison_val_list = []  # judge by each sentence of individual sample

    with jsonlines.open(input_file_path) as reader, jsonlines.open(output_file_path, mode='a') as writer:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            generated_answer = line["generated_answer"]
            
            # if generated_answer is none，skip
            if not generated_answer.strip():
                print(f"Skipping index {i} due to empty generated_answer.")
                continue
            
            generated_answer_sent = sent_tokenize(generated_answer)  # split into individual sentences
            answer = line["answer"]
            answer = truncate(tokenizer, answer)
            text = f"mednli: sentence1: {answer} sentence2: {generated_answer}"
            res = classify_text(model, tokenizer, text)
            line.update({"overall comparison": res[0] if res else "error"})  
            overall_comparison_val_list.append(res[1] if res else 0)  

            sent_val_list = []
            for sentence in generated_answer_sent:
                text = f"mednli: sentence1: {answer} sentence2: {sentence}"
                res = classify_text(model, tokenizer, text)
                if res is not None:
                    sent_val_list.append(res[1])  

            # check empty list
            if sent_val_list:
                sentence_comparison_val_list.append(np.mean(sent_val_list))
            else:
                print(f"Warning: sent_val_list is empty for index {i}")
                sentence_comparison_val_list.append(0)  

            writer.write(line)
    return [np.nanmean(overall_comparison_val_list), np.nanmean(sentence_comparison_val_list)]
