from CTRLEval.ctrleval import CTRLEval
import torch
import jsonlines
import numpy as np
from tqdm import tqdm

# This function is only for evaluating the score in consistency
def calc_ctrl_score(file_path):
    task = "topic"
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CTRLEval(iwf_dir='/home/ubuntu-user/Desktop/cognition-corrector-dev/Evaluate/CTRLEval/data/iwf_full.txt',
                        prompt_dir='/home/ubuntu-user/Desktop/cognition-corrector-dev/Evaluate/CTRLEval/data/prompt_{}.txt'.format(task),
                        verbal_dir='/home/ubuntu-user/Desktop/cognition-corrector-dev/Evaluate/CTRLEval/data/verbal_{}.txt'.format(task),
                        device=device)
    prefix_list = []
    data_list = []
    with jsonlines.open(file_path) as reader:
        reader = list(reader)
        for i, line in tqdm(enumerate(reader), total=len(reader)):
            question = line["question"]
            answer = line["answer"]
            generated_answer = line["generated_answer"]

            if not generated_answer.strip():
                print(f"Skipping index {i} due to empty generated_answer.")
                continue
            
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            generated_answer = generated_answer.replace("\n", " ")
            generated_answer = generated_answer.replace("\t", " ")

            data = question + generated_answer
            prefix_list.append(question)
            data_list.append(data)
    ctrl_score = scorer.score(aspect="cons",  data=data_list, prefix=prefix_list)

    return np.mean(ctrl_score)

