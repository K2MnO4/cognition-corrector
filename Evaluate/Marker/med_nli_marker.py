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
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")   
    model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI").to(device)

    overall_comparison_val_list = [] # judge the relationship between generated answer and golden answer by individual sample
    sentence_comparison_val_list = [] # judege by each sentence of individual sample

    with jsonlines.open(input_file_path) as reader, jsonlines.open(output_file_path, mode='w') as writer:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            generated_answer = line["generated_answer"]
            generated_answer_sent = sent_tokenize(generated_answer) # split into individual sentences
            answer = line["answer"]
            answer = truncate(tokenizer, answer)
            text = f"mednli: sentence1: {answer} sentence2: {generated_answer}"
            res = classify_text(model, tokenizer, text)
            line.update({"overall comparison": res[0]})
            overall_comparison_val_list.append(res[1])

            sent_val_list = []
            for i, sentence in enumerate(generated_answer_sent):
                text = f"mednli: sentence1: {answer} sentence2: {sentence}"
                res = classify_text(model, tokenizer, text)
                # line.update({f"sent_{i}_compare_answer": [sentence, res[0]]})
                sent_val_list.append(res[1])
            sentence_comparison_val_list.append(np.mean(sent_val_list))

            writer.write(line)
    return [np.mean(overall_comparison_val_list), np.mean(sentence_comparison_val_list)]
