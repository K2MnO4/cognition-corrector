# Judge the consistency between question and generated answer


from Scorer.CTRLEval.ctrleval import CTRLEval
import torch
import numpy as np

# This function is only for evaluating the score in consistency
def get_ctrl_score(knowledge, answer):
    task = "topic"
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CTRLEval(iwf_dir=f'Core/Scorer/CTRLEval/data/iwf_full.txt',
                        prompt_dir=f'Core/Scorer/CTRLEval/data/prompt_{task}.txt',
                        verbal_dir=f'Core/Scorer/CTRLEval/data/verbal_{task}.txt',
                        device=device)
    prefix_list = []
    data_list = []
    knowledge = knowledge.replace("\n", " ")
    knowledge = knowledge.replace("\t", " ")
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    prefix_list.append(knowledge)
    data = knowledge + answer
    data_list.append(data)

    ctrl_score = scorer.score(aspect="cons",  data=data_list, prefix=prefix_list)
    return np.nanmean(ctrl_score)