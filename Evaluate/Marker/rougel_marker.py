from rouge import Rouge

import jsonlines
import numpy as np
from tqdm import tqdm

def calc_rl_score(file_path):
    r_l_score_list = []
    with jsonlines.open(file_path) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            answer = line.get("answer", "").replace("\n", " ").replace("\t", " ")
            generated_answer = line.get("generated_answer", "").replace("\n", " ").replace("\t", " ")

            # Check if answer and generated_answer are empty
            if not generated_answer:
                print(f"Warning: Generated answer is empty at line {i}, skipping this entry.")
                continue
            if not answer:
                print(f"Warning: Reference answer is empty at line {i}, skipping this entry.")
                continue

            # caculate ROUGE 
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=generated_answer, refs=answer)
            r_l_score_list.append(rouge_score[0]["rouge-l"]["f"])

    # Calculate the mean of the ROUGE-L scores
    return np.nanmean(r_l_score_list) if r_l_score_list else 0
