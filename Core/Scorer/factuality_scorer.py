from .opt_score import OPTScorer



def get_factuality_score(gptscore_model_name, question, bg_knowlege, demo_num = 0):
    PREFIX = {
        0: f'''Based on the Question, please generate the factual response. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.
    Question: {question}
    Response: ''', }
    prefix = PREFIX[demo_num]

    srcs = [prefix]
    tgts = [bg_knowlege]

    check_point = "facebook/opt-350m" # default gpt-score model
    if gptscore_model_name == "opt-125m":
        check_point = "facebook/opt-125m"
    elif gptscore_model_name == "opt-1.3b":
        check_point = "facebook/opt-1.3b"
    elif gptscore_model_name == "gpt2-medium":
        check_point = "openai-community/gpt2-medium"
    elif gptscore_model_name == "gpt2-large":
          check_point = "openai-community/gpt2-large"

    score_list = OPTScorer(device='cuda', max_length=1024, checkpoint=check_point).score(srcs, tgts, prompt_text="", batch_size=1)
    gptscore = score_list[0]
    return gptscore