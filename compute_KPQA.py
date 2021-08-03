import numpy as np
import pickle
import json
import os
import kpqa_coco
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer
from collections import defaultdict
from kpqa_bertscore.bert_score.utils import get_idf_dict 
from kpqa_bertscore.bert_score import score
from kpqa_bertscore.bert_score import BERTScorer
from copy import copy

import transformers
from transformers import (AutoTokenizer, BertForTokenClassification, AutoConfig)
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

import string
import argparse
import kpw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_kp', type=bool, default=True)
    parser.add_argument('--is_q', type=int, default=1)
    parser.add_argument('--data', type=str, default='sample.csv')
    parser.add_argument('--model_path',type=str, default='ckpt')
    parser.add_argument('--num_ref', type=int, default=1)
    parser.add_argument('--out_file', type=str, default='results.csv')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    ## Configuration
    stopwords = []

    is_q = bool(args.is_q)

    # Number of References for Each Dataset
    Nr = args.num_ref

    use_idf = True
    use_stopwords = False
    use_KP = True
    
    if not use_stopwords:
        stopwords=None

    ## Data Load
    data = pd.read_csv(args.data)    

    refs = []
    for i in range(Nr):
        refs.append(data['reference'+str(i+1)])
    ques = data['question']
    hyps = data['answer']

    all_refs = []
    for x in refs:
        all_refs += list(x)        

    ## KPW Load
    PATH = os.path.join(args.model_path)
    config = AutoConfig.from_pretrained(os.path.join(PATH, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = BertForTokenClassification.from_pretrained(PATH)    
    model = model.cuda()
    
    probs = kpw.get_kpw(model, hyps, refs, ques, is_q=is_q)        
        
    ## BERTScorer Setting
    # model_type = 'bert-base-uncased' # shows slight worse performance
    model_type = 'bert-large-uncased-whole-word-masking-finetuned-squad'

    model_path = None
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    n_layers = 1

    scorer = BERTScorer(lang="en", model_type=model_type,
                        num_layers=n_layers, idf=use_idf,
                        idf_sents=all_refs, stopwords=stopwords,
                        rescale_with_baseline=False,
                       model_path=model_path)
    
    scorer_o = BERTScorer(lang="en", model_type=model_type,
                        num_layers=n_layers, idf=True,
                        idf_sents=all_refs, stopwords=None,
                        rescale_with_baseline=False,
                       model_path=model_path)

    ## BERTScore Calculation
    pscores = []
    rscores = []
    f1scores = []
    f1scores_o = []

    for j in tqdm(range(Nr)):

        reference = list(refs[j])
        hypothesis = list(hyps)

        P, R, F1 =  scorer.score(hypothesis, reference, w_hyps=probs[0], w_refs=probs[j+1])
        P = P.numpy()
        R = R.numpy()
        F1 = F1.numpy()
    
        P_, R_, F1_ = scorer_o.score(hypothesis, reference)
        f1scores_o.append(F1_.numpy())
       
        pscores.append(P)
        rscores.append(R)
        f1scores.append(F1)        
  
    f1_max_o = np.max(f1scores_o, axis=0)
    p_max = np.max(pscores, axis=0)
    r_max = np.max(rscores, axis=0)
    f1_max = np.max(f1scores, axis=0)
    
    ## Exact Match Based Metrics
    coco_evaluator_c = kpqa_coco.CocoEvaluator(average=False)
    rouge_kpqa_evaluator = kpqa_coco.RougeEvaluator(num_parallel_calls=1, average=False)
    
    r1s_k = []
    rs_k = []
    b1s_k = []
   
    mode = 0

    for j in tqdm(range(Nr)):

        reference = list(refs[j])
        hypothesis = list(hyps)

        if(mode):
            hypothesis_ = [rm_stop(x, stopwords) for x in hypothesis]
            reference_ = [rm_stop(x, stopwords) for x in reference]
            rouge_eval = rouge_kpqa_evaluator.run_evaluation(hypothesis_, reference_, probs[0], probs[j+1])
             
        else:
            rouge_eval = rouge_kpqa_evaluator.run_evaluation(hypothesis, reference, probs[0], probs[j+1])
            coco_eval_c = coco_evaluator_c.run_evaluation(hypothesis, reference, probs[0], probs[j+1])

        r1s_k.append([x for x in rouge_eval['rouge1'][0]])
        rs_k.append([x for x in rouge_eval['rougeL'][0]])
        b1s_k.append([x['Bleu_1'] for x in coco_eval_c])

    rs_k_max = np.max(rs_k, axis = 0)
    b1s_k_max = np.max(b1s_k, axis = 0)
    r1s_k_max = np.max(r1s_k, axis = 0)
    
    ## Compute Metric
    lst = [f1_max_o, b1s_k_max, rs_k_max, f1_max]
    average_scores = [np.average(x) for x in lst]
    df = pd.DataFrame(lst, index=['BERTScore','BLEU-1-KPQA', 'ROUGE-L-KPQA', 'BERTScore-KPQA']).T
    df_avg = pd.DataFrame(average_scores, index=['BERTScore','BLEU-1-KPQA', 'ROUGE-L-KPQA', 'BERTScore-KPQA']).T

    print("## Result")
    print(df_avg)
      
    df.to_csv(args.out_file, index=False)
