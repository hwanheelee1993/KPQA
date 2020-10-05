__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from tqdm import tqdm
import nltk
from transformers import AutoTokenizer
model_type = 'bert-base-uncased'

_COCO_TYPE_TO_METRIC = {
    "BLEU": (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    "ROUGE_L": (Rouge(), "ROUGE_L"),
    "CIDEr": (Cider(), "CIDEr"),
}

class COCOEvalCap:
    def __init__(self, coco, cocoRes, cocoTypes, tokenization_fn=None):       
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}
        self.cocoTypes = cocoTypes
        self.cocoTypes = ["BLEU", "ROUGE_L"]
        self.tokenization_fn = tokenization_fn
    
    def _tokenize(self, caps):
        #tokenizer = AutoTokenizer.from_pretrained(model_type)

        caps_ = dict()
        for i in range(len(caps)):
            caps_[i] = [caps[i][0]['caption']]
        return caps_
    
    def evaluate(self, pred_kp=None, ans_kp=None):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        #print('tokenization...')
        tokenizer = PTBTokenizer(self.tokenization_fn)

        gts = self._tokenize(gts)
        res = self._tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        #print('setting up scorers...')
        scorers = [_COCO_TYPE_TO_METRIC[coco_type] for coco_type in self.cocoTypes]

        # =================================================
        # Compute scores
        # =================================================
        all_scores = []
        for scorer, method in tqdm(scorers):
            #print('computing {} score...'.format(scorer.method()))
            score, scores = scorer.compute_score(gts, res, ans_kp=ans_kp, pred_kp=pred_kp)
            #all_scores.append(scores)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    #print("{}: {:3}".format(m, sc))
                    
            else:
                #print("%", method)
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                #print("{}: {:3}".format(method, score))
        self.setEvalImgs()
        return self.evalImgs

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
