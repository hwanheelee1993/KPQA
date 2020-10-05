import numpy as np
import torch
import transformers
import os
from tqdm import tqdm
from torch.nn.functional import softmax
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from transformers import (AutoModel, GPT2Tokenizer,
AutoTokenizer, AutoModelForSequenceClassification, BertForTokenClassification, AutoConfig, RobertaConfig)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def keyword(output, input_ids, ans_s, ans_e):
    output_ = (-(torch.argmax(output[0], axis=2)-1)*input_ids).cpu().numpy()
    key_ids = output_[0][ans_s:ans_e]
    keywords = tokenizer.convert_ids_to_tokens(key_ids)
    keywords = [x for x in keywords if x != '[PAD]']
    return keywords

def model_output(model, hyps, refs, ques, idx, stopwords=None, is_print=False, is_q=True):
    scores = []
    keylist = []
    problist = []
    model = model.eval()
    
    for i in range(len(hyps)):   
        with torch.no_grad():
            hyp = hyps[i]
            
            if(idx==-1):
                ans = hyp
            else:
                ans = refs[idx][i]

            if(is_q):
                input_sents = ques[i]+' [SEP] '+ans
            else:
                input_sents = ans

            input_ids_enc = tokenizer.encode(input_sents)
            input_ids = torch.tensor([input_ids_enc])
            sep_idx = input_ids_enc.index(102)    
            input_ids = input_ids.cuda()
            
            seg_ids = torch.zeros_like(input_ids)
            
            if(is_q):
                seg_ids = torch.zeros_like(input_ids)
                
                seg_ids[0][sep_idx+1:] = 1
                seg_ids[0][0] = 1
                
            inputs = {"input_ids": input_ids, 'token_type_ids': seg_ids}
            output = model(**inputs)
            
            probs = softmax(output[0],dim=2)[0,:,0].cpu().numpy()
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            if(is_q):
                ans_s = input_tokens.index('[SEP]')+1
                ans_e = len(input_tokens)-1
            else:
                ans_s = 1
                ans_e = len(input_tokens)-1
                
            probs = probs[ans_s:ans_e]
                        
            problist.append(probs)                  
            
            key = keyword(output, input_ids, ans_s, ans_e)
            keylist.append(tokenizer.convert_tokens_to_string(key))
                        
    return keylist, problist

def get_kpw(model, HYPS, REFS, QUES, is_q=True):
    probs = []
    Nr = len(REFS)
    for e in range(Nr+1):
        _, problist = model_output(model, HYPS, REFS, QUES, e-1, is_print=False, is_q=is_q)
        probs.append(problist)
        
    return probs
        
        
        
        
