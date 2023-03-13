# WIGRAPH
This repo contains code for the paper ``Improving Interpretability via Explicit Word Interaction Graph Layer`` published in AAAI 2023. 

Recent NLP literature has seen growing interest in improving model interpretability. Along this direction, we propose a trainable neural network layer that learns a global interaction graph between words and then selects more informative words using the learned word interactions. Our layer, we call WIGRAPH, can plug into any neural network-based NLP text classifiers right after its word embedding layer. Adding the WIGRAPH layer substantially improves NLP models' interpretability and enhances models' prediction performance at the same time.

``
@article{sekhon2023improving,
  title={Improving Interpretability via Explicit Word Interaction Graph Layer},
  author={Sekhon, Arshdeep and Chen, Hanjie and Shrivastava, Aman and Wang, Zhe and Ji, Yangfeng and Qi, Yanjun},
  journal={arXiv preprint arXiv:2302.02016},
  year={2023}
}
``

## Requirements 
- Pytorch==1.5.1
- Transformers==3.3.0
- datasets=1.1.2
## Prerequisites

The `data` folder contains the dataset that are used for WIGRAPH. 
The datasets used can be downloaded from : [datasets link](https://drive.google.com/file/d/1id1F7N9vXbpL3Y8Omhq2zosT_MTxOT8A/view?usp=share_link)

The `metadata` folder contains precomputed counts of words and interactions and are used to extract a subset of words and interactions used to learn the model. These can be downloaded from [metadata link(precomputed counts)](https://drive.google.com/file/d/1CDQUYJZ7CV_33OU9Or-Q17E2wiCh6o0z/view?usp=share_link).

The models used here as classifiers can be downloaded from : [finetuned models](https://drive.google.com/file/d/1id1F7N9vXbpL3Y8Omhq2zosT_MTxOT8A/view?usp=share_link)

## Training 

The wigraph layer that can be plugged in front of any NLP classifier can be found in `imask.py`. 

To train a model using WIGRAPH:
``
main.py --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 --task-name sst2 --learning_rate 1e-05 --factor 1000.0 --mask-hidden-dim 32 --backbone distilbert --save-dir results/distilbert/ --init-mode static --beta-i 0.0 --beta-g 1.0 --beta-s 1.0 --imask-dropout 0.3 --project WIGRAPH --non-linearity gelu --num_train_epochs 10 --seed 42 --onlyA --max_sent_len 56 --anneal
``

The relevant hyperparameters are 
1. task-name : [sst2, sst1, trec, subj, imdb, agnews]
2. backbone : [distilbert, roberta, bert] 
3. beta-i : 0.0 for WIGRAPH-A, 1.0 for WIGRAPH-A-R
4. beta-s : sparsity weight
5. onlyA : does not train R, enable for WIGRAPH-A
5. max_sent_len : [sst2(56), sst1(56), trec(32), subj(128), imdb(512), agnews(128)]

## Interpretation

- We provide interpretation using SHAP as well as LIME. A WIGRAPH model can be interpreted using the following:

 ``
interpret.py --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 --task-name sst2 --learning_rate 1e-05 --factor 1000.0 --mask-hidden-dim 32 --backbone distilbert  --save-dir results/distilbert/ --init-mode static --beta-i 0.0 --beta-g 1.0 --beta-s 1.0 --imask-dropout 0.3 --project WIGRAPH --non-linearity gelu --num_train_epochs 10 --seed 42 --onlyA --max_sent_len 56 --no-save --anneal
``

- The aopc (after interpretation files are generated) can be measured using:

``
aopc.py --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 --task-name sst2 --learning_rate 1e-05 --factor 1000.0 --mask-hidden-dim 32 --backbone distilbert  --save-dir results/distilbert/ --init-mode static --beta-i 0.0 --beta-g 1.0 --beta-s 1.0 --imask-dropout 0.3 --project WIGRAPH --non-linearity gelu --num_train_epochs 10 --seed 42 --onlyA --max_sent_len 56 --no-save --anneal
``

- The hyperparameters for both the above are the same as the training hyperparameters.
