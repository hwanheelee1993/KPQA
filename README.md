# KPQA

This repository provides an evaluation metric for generative question answering systems based on our NAACL 2021 paper [KPQA: A Metric for Generative Question Answering Using Keyphrase Weights](https://www.aclweb.org/anthology/2021.naacl-main.170.pdf). <br> Here, we provide the code to compute KPQA-metric, and human annotated data.

<h2> Usage </h2>

<h3> 1. Install Prerequisites </h3>

Create a python 3.6 environment and then install the requirements.


Install packages using "requirements.txt"

```
conda create -name kpqa python=3.6
pip install -r requirements.txt
```

<h3> 2. Download Pretrained Model </h3>
We provide the pre-trained KPQA model in the following link. <br>
https://drive.google.com/file/d/1pHQuPhf-LBFTBRabjIeTpKy3KGlMtyzT/view?usp=sharing <br>
Download the "ckpt.zip" and extract it. (default directory is "./ckpt")

<h3> 3. Compute Metric </h3>
You can compute KPQA-metric using "compute_KPQA.py" as follows. <br><br>

```
python compute_correlation.py \
  --data sample.csv \ # Target data to compute the score. Please see the "sample.csv" for file format
  --model_path $CHECKPOINT_DIR \ # Path of checkpoint directory (extract path of "ckpt.zip")
  --out_file results.csv \ # output file that has score for each question-answer pair. Please see the the sample result in "result.csv".
  --num_ref 1 \ # For usage in computing the score with multiple references.
```

<h3> Train KPQA (optional) </h3>
You can train your own KPQA model using the provided dataset or your own dataset using "train.py". (script for running with the default setting is "train_kpqa.sh") <br>

<h2> Dataset </h2>
We provide human judgments of correctness for 4 datasets:MS-MARCO NLG, AVSD, Narrative QA and SemEval 2018 Task 11 (SemEval). <br>
For MS-MARCO NLG and AVSD, we generate the answer using two models for each dataset.

For NarrativeQA and SemEval, we preprocessed the dataset from [Evaluating Question Answering Evaluation](https://www.aclweb.org/anthology/D19-5817). <br>

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
