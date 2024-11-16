from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class CTRLEval:
    def __init__(self, iwf_dir=None, prompt_dir=None, verbal_dir=None,
                 device='cuda', model_name_or_path='google/pegasus-large'):
        # Load inverse word frequency (IWF) scores
        with open(iwf_dir, 'r') as f_iwf:
            self.iwf_score = [float(line.strip()) for line in f_iwf.readlines()]

        # Load prompts and verbalizers for attribute relevance
        with open(prompt_dir, 'r') as f_pr:
            self.prompt_list = [line.strip() for line in f_pr.readlines()]
        
        self.verbal_list = []
        with open(verbal_dir, 'r') as f_veb:
            lines = f_veb.readlines()
            self.label_name = lines[0].strip().split('\t')  # First line: label names
            for line in lines[1:]:
                self.verbal_list.append(line.strip().split('\t'))  # Following lines: verbalizers

        # Set device and model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name_or_path)
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.model.config.pad_token_id)

    def lm_score(self, src_text, tgt_text, has_iwf=True, add_special_tokens=True):
        # Tokenize source and target text
        batch = self.tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        labels = self.tokenizer(tgt_text, truncation=True, padding='longest', add_special_tokens=add_special_tokens,
                                return_tensors="pt").to(self.device)

        # Check if labels are empty
        if labels['input_ids'].size(1) == 0:
            print("Warning: labels['input_ids'] is empty, skipping this entry.")
            return None, None

        # Compute IWF-based scores if applicable
        tgt_score = []
        if has_iwf:
            for label_id in range(labels['input_ids'].shape[0]):
                token_ids = labels['input_ids'][label_id].cpu().numpy()
                iwf_scores = [self.iwf_score[token_id] if token_id < len(self.iwf_score) else 0 for token_id in token_ids]
                tgt_score.append(max(iwf_scores) if iwf_scores else 0)

        # Calculate loss
        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels['input_ids'])
        logits = output.logits.view(-1, self.model.config.vocab_size)
        loss = self.loss_fct(logits, labels['input_ids'].view(-1))

        tgt_len = labels['attention_mask'].sum(dim=1)
        loss = loss.view(labels['input_ids'].shape[0], -1)
        loss = loss.sum(dim=1) / tgt_len

        return loss, tgt_score

    def coh_score(self, data, batch_size):
        data_split = [sent_tokenize(data_ele) for data_ele in data]

        def get_mask_data(data_list):
            src_list, tgt_list, len_list = [], [], []
            for data_ele in data_list:
                src_list_ele, tgt_list_ele = [], []
                for idx in range(len(data_ele)):
                    tgt_list_ele.append(data_ele[idx])
                    src_list_ele.append(' '.join(data_ele[:idx]) + ' <mask_1> ' + ' '.join(data_ele[idx + 1:]))
                src_list.extend(src_list_ele)
                tgt_list.extend(tgt_list_ele)
                len_list.append(len(data_ele))
            return src_list, tgt_list, len_list

        src_data, tgt_data, data_len = get_mask_data(data_split)
        eval_score, beta = [], []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, tgt_score = self.lm_score(src_text, tgt_text)
                if loss is None or tgt_score is None:
                    continue
                cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]
            eval_score.extend(cur_score)
            beta.extend(tgt_score)

        res_score = self._compute_weighted_score(eval_score, beta, data_len)
        return res_score

    def cons_score(self, data, prefix, batch_size):
        def get_mask_data(data_list, prefix_list):
            src_list, tgt_list, len_list = [], [], []
            for data_ele, prefix_ele in zip(data_list, prefix_list):
                if not data_ele.startswith(prefix_ele):
                    print(f"Warning: '{prefix_ele}' is not a prefix of '{data_ele}', skipping this entry.")
                    continue
                src_list.extend([prefix_ele + ' <mask_1>', '<mask_1> ' + data_ele[len(prefix_ele):]])
                tgt_list.extend([data_ele[len(prefix_ele):], prefix_ele])
                len_list.append(2)
            return src_list, tgt_list, len_list

        src_data, tgt_data, data_len = get_mask_data(data, prefix)
        eval_score, beta = [], []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, tgt_score = self.lm_score(src_text, tgt_text, add_special_tokens=False)
                if loss is None or tgt_score is None:
                    continue
                cur_score = [-loss_ele.detach().cpu().numpy() for loss_ele in loss]
            eval_score.extend(cur_score)
            beta.extend(tgt_score)

        res_score = self._compute_weighted_score(eval_score, beta, data_len)
        return res_score

    def ar_score(self, data, label_str, batch_size):
        label = [self.label_name.index(label_ele) for label_ele in label_str]

        def get_mask_data(data_list, prompt_list, verbal_list):
            src_list, tgt_list = [], []
            for data_ele in data_list:
                for idx in range(len(prompt_list)):
                    for idy in range(len(verbal_list)):
                        for idz in range(len(verbal_list[0])):
                            src_list.append(prompt_list[idx].replace('<gen_result>', data_ele).replace('<mask_token>', '<mask_1>'))
                            tgt_list.append(verbal_list[idy][idz])
            return src_list, tgt_list

        src_data, tgt_data = get_mask_data(data, self.prompt_list, self.verbal_list)
        eval_score = []
        for data_id in tqdm(range(0, len(src_data), batch_size)):
            src_text, tgt_text = src_data[data_id: data_id + batch_size], tgt_data[data_id: data_id + batch_size]
            self.model.eval()
            with torch.no_grad():
                loss, _ = self.lm_score(src_text, tgt_text, has_iwf=False, add_special_tokens=False)
                eval_score.extend([torch.exp(-loss_ele).detach().cpu().numpy() for loss_ele in loss])

        score_pair = np.reshape(eval_score, (-1, len(self.verbal_list[0])))
        weight_unnormal = np.sum(score_pair, axis=1)
        score_pair /= np.sum(score_pair, axis=1, keepdims=True)
        score_data = np.reshape(score_pair, (-1, len(self.prompt_list) * len(self.verbal_list), len(self.verbal_list[0])))
        weight_normal = np.expand_dims(weight_unnormal / np.sum(weight_unnormal, axis=1, keepdims=True), axis=2)
        res_score = np.choose(np.array(label), np.sum(score_data * weight_normal, axis=1).T)

        return res_score

    def _compute_weighted_score(self, eval_score, beta, data_len):
        res_score = []
        data_st = 0
        for len_ele in data_len:
            segment_eval_score = eval_score[data_st: data_st + len_ele]
            segment_beta = beta[data_st: data_st + len_ele]
            
            # Ensure segment_beta and segment_eval_score are non-empty
            if not segment_beta or not segment_eval_score:
                print(f"Warning: Empty eval_score or beta segment at index {data_st}, appending 0 as score.")
                res_score.append(0)
            elif sum(segment_beta) > 0:
                res_score.append(np.dot(segment_eval_score, segment_beta) / sum(segment_beta))
            else:
                res_score.append(np.mean(segment_eval_score))
            
            data_st += len_ele
        return res_score

    def score(self, aspect, data, prefix=None, label=None, batch_size=1):
        if aspect == 'coh':
            return self.coh_score(data, batch_size)
        elif aspect == 'cons':
            return self.cons_score(data, prefix, batch_size)
        elif aspect == 'ar':
            return self.ar_score(data, label, batch_size)
