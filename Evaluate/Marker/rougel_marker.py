from rouge import Rouge

import jsonlines
import numpy as np
from tqdm import tqdm


def calc_rl_score(file_path):
    r_l_score_list = []

    with jsonlines.open(file_path) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            answer = line["answer"]
            generated_answer = line["generated_answer"]

            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")

            generated_answer = generated_answer.replace("\n", " ")
            generated_answer = generated_answer.replace("\t", " ")
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=generated_answer, refs=answer)
            r_l_score_list.append(rouge_score[0]["rouge-l"]["f"])

    return np.mean(r_l_score_list)