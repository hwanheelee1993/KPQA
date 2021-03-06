# KPQA

This repository provides an evaluation metric for generative question answering systems based on our NAACL 2021 paper [KPQA: A Metric for Generative Question Answering Using Keyphrase Weights](https://www.aclweb.org/anthology/2021.naacl-main.170.pdf). <br> Here, we provide the code to train KPQA, pretrained model, human annotated data and the code to compute KPQA-metric.

The repository will soon be updated until 6/10 in a more useful form using demo in jupyter notebook.(weights will be uploaded to huggingface models)

<h2> Dataset </h2>
We provide human judgments of correctness for 4 datasets:MS-MARCO NLG, AVSD, Narrative QA and SemEval 2018 Task 11 (SemEval). <br>
For MS-MARCO NLG and AVSD, we generate the answer using two models for each dataset.
For NarrativeQA and SemEval, we preprocessed the dataset from [Evaluating Question Answering Evaluation](https://www.aclweb.org/anthology/D19-5817). <br>

<h2> Usage </h2>

<h3> 1. Install Prerequisites </h3>
Install packages using "requirements.txt"

<h3> 2. Download Pretrained Model </h3>
We provide the pre-trained KPQA model in the following link. <br>
https://drive.google.com/file/d/1pHQuPhf-LBFTBRabjIeTpKy3KGlMtyzT/view?usp=sharing <br>
Download the "ckpt.zip" and extract it.

<h3> 3. Compute Metric </h3>
You can compute KPQA-metric using "compute_correlation.py" <br><br>

python compute_correlation.py \ <br>
  --dataset marco \ # Target dataset to evaluate the metric <br>
  --qa_model unilm \ # The model used to generate answer. <br>
  --model_dir $CHECKPOINT_DIR \ # Path of checkpoint directory (extract path of "ckpt.zip") <br><br>

For evaluating various metrics on MS-MARCO NLG dataset, the printed result (correlation with human judgments) will be as follows. <br><br>
Metrics         | Pearson    | Spearman <br>
BLEU-1          | 0.369      | 0.337 <br>
BLEU-4          | 0.173      | 0.224 <br>
ROUGE-L         | 0.317      | 0.289 <br>
CIDEr           | 0.261      | 0.256 <br>
BERTScore       | 0.469      | 0.445 <br>
BLEU-1-KPQA     | 0.729      | 0.676 <br>
ROUGE-L-KPQA    | 0.735      | 0.674 <br>
BERTScore-KPQA  | 0.698      | 0.66 <br><br>

<h3> Train KPQA (optional) </h3>
You can train your own KPQA model using the provided dataset or your own dataset using "train.py".<br>
You can train using the default setting with "train_kpqa.sh"

## Reference

If you find this repo useful, please consider citing:

```
@inproceedings{lee2021kpqa,
  title={KPQA: A Metric for Generative Question Answering Using Keyphrase Weights},
  author={Lee, Hwanhee and Yoon, Seunghyun and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Shin, Joongbo and Jung, Kyomin},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={2105--2115},
  year={2021}
}
```
