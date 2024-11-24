import jsonlines
import numpy as np
from tqdm import tqdm
from collections import Counter

def calc_f1_score(file_path):
    f1_score_list = []

    with jsonlines.open(file_path) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            answer = line["answer"]
            if isinstance(answer, list):
                answer = answer[0]
            generated_answer = line["generated_answer"]

            if not generated_answer.strip():
                print(f"Skipping index {i} due to empty generated_answer.")
                continue
            
            answer = answer.replace(".", " ")
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            generated_answer = generated_answer.replace(".", " ")
            generated_answer = generated_answer.replace("\n", " ")
            generated_answer = generated_answer.replace("\t", " ")

            reference_tokens = answer.split()
            hypothesis_tokens = generated_answer.split()

            common = Counter(reference_tokens) & Counter(hypothesis_tokens)
            overlap_num = sum(common.values())
            golden_num = len(reference_tokens)
            generated_num = len(hypothesis_tokens)

            precision = (overlap_num + 1) / (generated_num + 1)
            recall = (overlap_num + 1) / (golden_num + 1)
            f1_score = 2 * (precision * recall) / (precision + recall)

            f1_score_list.append(f1_score)

    return np.nanmean(f1_score_list)
