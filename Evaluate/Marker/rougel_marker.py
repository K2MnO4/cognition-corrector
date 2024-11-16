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

            # 检查 answer 和 generated_answer 是否为空
            if not generated_answer:
                print(f"Warning: Generated answer is empty at line {i}, skipping this entry.")
                continue
            if not answer:
                print(f"Warning: Reference answer is empty at line {i}, skipping this entry.")
                continue

            # 计算 ROUGE 分数
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=generated_answer, refs=answer)
            r_l_score_list.append(rouge_score[0]["rouge-l"]["f"])

    # 计算 ROUGE-L 分数的平均值
    return np.mean(r_l_score_list) if r_l_score_list else 0
