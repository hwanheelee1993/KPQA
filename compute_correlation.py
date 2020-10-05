import numpy as np
import pickle
import json
import os
import language_evaluation
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
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import nltk
from transformers import (AutoTokenizer, BertForTokenClassification, AutoConfig)

from nltk.tokenize import word_tokenize
import string
import argparse
import kpw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='marco')
    parser.add_argument('--use_kp', type=bool, default=True)
    parser.add_argument('--model_dir', type=str, default='ckpt')
    parser.add_argument('--is_q', type=int, default=1)
    parser.add_argument('--qa_model', type=str, default='unilm')
    parser.add_argument('--datadir', type=str, default='data/human_eval')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    ## Configuration
    stopwords = []

    dataset = args.dataset
    qa_model = args.qa_model
    is_q = bool(args.is_q)

    if(dataset == 'avsd'):    
        Nr = 5
    elif(dataset == 'nrqa' or dataset == 'semeval'):
        Nr = 2
    else:
        Nr = 1
    
    use_idf = True
    use_stopwords = False
    use_KP = True
    
    if not use_stopwords:
        stopwords=None

    ## Human Eval Load
    datadir = args.datadir
    df_fname = os.path.join(datadir, dataset+'_'+qa_model+'.csv')
    human_eval = pd.read_csv(df_fname)    

    refs = []
    for i in range(Nr):
        refs.append(human_eval['refs'+str(i+1)])
    ques = human_eval['query']
    hyps = human_eval['hyps']

    scores = human_eval['scores']

    all_refs = []
    for x in refs:
        all_refs += list(x)        

    ## KPW Load
    PATH = os.path.join(args.model_dir)
    config = AutoConfig.from_pretrained(os.path.join(PATH, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = BertForTokenClassification.from_pretrained(PATH)    
    model = model.cuda()
    
    probs = kpw.get_kpw(model, hyps, refs, ques, is_q=is_q)        
        
    ## BERTScorer Setting
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
    coco_evaluator = language_evaluation.CocoEvaluator(average=False)
    coco_evaluator_c = kpqa_coco.CocoEvaluator(average=False)
    rouge_kpqa_evaluator = kpqa_coco.RougeEvaluator(num_parallel_calls=1, average=False)
    
    b1s = []
    b4s = []
    rs = []
    ms = []
    cs = []
    
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
            coco_eval = coco_evaluator.run_evaluation(hypothesis_, reference_, probs[0], probs[j+1])
            rouge_eval = rouge_kpqa_evaluator.run_evaluation(hypothesis_, reference_, probs[0], probs[j+1])
             
        else:
            coco_eval = coco_evaluator.run_evaluation(hypothesis, reference)
            rouge_eval = rouge_kpqa_evaluator.run_evaluation(hypothesis, reference, probs[0], probs[j+1])
            coco_eval_c = coco_evaluator_c.run_evaluation(hypothesis, reference, probs[0], probs[j+1])

        b1s.append([x['Bleu_1'] for x in coco_eval])
        rs.append([x['ROUGE_L'] for x in coco_eval])
        cs.append([x['CIDEr'] for x in coco_eval])
        r1s_k.append([x for x in rouge_eval['rouge1'][0]])
        rs_k.append([x for x in rouge_eval['rougeL'][0]])
        ms.append([x['METEOR'] for x in coco_eval])
        b1s_k.append([x['Bleu_1'] for x in coco_eval_c])
        b4s.append([x['Bleu_4'] for x in coco_eval])

    b1s_max = np.max(b1s, axis = 0)
    rs_max = np.max(rs, axis = 0)
    ms_max = np.max(ms, axis = 0)
    rs_k_max = np.max(rs_k, axis = 0)
    b1s_k_max = np.max(b1s_k, axis = 0)
    b4s_max = np.max(b4s, axis = 0)
    cs_max = np.max(cs, axis = 0)
    r1s_k_max = np.max(r1s_k, axis = 0)
    
    ## Compute Correlation
    lst = [list(scores),  b1s_max, b4s_max, rs_max, cs_max, f1_max_o, b1s_k_max, rs_k_max, f1_max]
    df = pd.DataFrame(lst, index=['Human',  'BLEU-1','BLEU-4', 'ROUGE-L', 'CIDEr','BERTScore','BLEU-1-KPQA', 'ROUGE-L-KPQA', 'BERTScore-KPQA']).T

    corr_p = df.corr(method='pearson')
    corr_k = df.corr(method='kendall')
    corr_s = df.corr(method='spearman')    
    
    name = list(corr_p.head(1))
    corr_p = list(corr_p.iloc[0])
    corr_s = list(corr_s.iloc[0])    
    
    print("# Evaluation Results for %s")
    print('{:<15}'.format('Metrics'), '|', '{:<10}'.format('Pearson'), '|', '{:<10}'.format('Spearman')) 
    for w in range(1, len(name)):
        print('{:<15}'.format(name[w]), '|', '{:<10}'.format(round(corr_p[w], 3)), '|', '{:<10}'.format(round(corr_s[w], 3))) 
        
    df.to_csv(args.dataset+'_'+args.qa_model+'.csv')
