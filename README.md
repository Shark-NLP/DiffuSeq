# <img src="img/logo.jpg" width="8%" alt="" align=center /> DiffuSeq

Official Codebase for [*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*](https://arxiv.org/abs/2210.08933).

<p align = "center">
<img src="img/diffuseq-process.png" width="95%" alt="" align=center />
</p>
<p align = "center">
The diffusion process of our conditional diffusion language model DiffuSeq.
</p>

## Highlights
- Our proposed __*DiffuSeq*__ as a conditional language model is trained end-to-end in a classifier-free manner.
- We establish a theoretical
connection among AR, NAR and __*DiffuSeq*__ models (refer to our original paper).
- __*DiffuSeq*__ is a powerful model for text
generation, matching or even surpassing competitive AR, iterative NAR,
and large-PLMs on quality and diversity.

Our study addresses promising achievements by such a new
sequence-to-sequence learning paradigm.

<p align = "center">
<img src="img/result-1.png" width="80%" alt="" align=center />
</p>
<p align = "center">
<img src="img/result-2.png" width=80%" alt="" align=center />
</p>



## Setup:
The code is based on PyTorch and HuggingFace `transformers`.
```bash 
pip install -r requirements.txt 
```

## Datasets
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task | Datasets | Training Samples | Source | Used in __*DiffuSeq*__
|-|-|-|-|-|
| Open-domain Dialogue | Commonsense Conversation | 3382k | [CCM](https://github.com/thu-coai/ccm) | [download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp?usp=sharing) |
| Question Generation | Quasar-T | 117k | [OpenQA](https://github.com/thunlp/OpenQA) | [download](https://drive.google.com/drive/folders/122YK0IElSnGZbPMigXrduTVL1geB4wEW?usp=sharing) |
| Text Simplification | Wiki-alignment | 677k | [Wiki-auto](https://github.com/chaojiang06/wiki-auto) | [download](https://drive.google.com/drive/folders/1BlWtD1UbnL_ef06Riq-gABlL0Zb50s-d?usp=sharing)|
| Paraphrase | QQP | 144k|  [Kaggle](https://www.kaggle.com/c/quora-question-pairs) | [download](https://drive.google.com/drive/folders/1BHGCeHRZU7MQF3rsqXBIOCU2WIC3W6fb?usp=sharing) |

## DiffuSeq Training
```bash
cd scripts
bash train.sh
```
Arguments explanation:
- ```--dataset```: the name of datasets, just for notation
- ```--data_dir```: the path to the saved datasets folder, containing ```train.jsonl,test.jsonl,valid.jsonl```
- ```--seq_len```: the max length of sequence $z$ ($x\oplus y$)
- ```--resume_checkpoint```: if not none, restore this checkpoint and continue training
- ```--vocab```: the tokenizer is initialized using bert or load your own preprocessed vocab dictionary (e.g. using BPE)

It will take 2 more days to train a __*DiffuSeq*__ model on 4 NVIDIA A100 80G GPUs for QG and QQP, and the training steps should be increased accordingly along with the size of the training set. To reproduce the results of Table 1 in our paper, we suggest the following configuration for each dataset when training.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset qqp --data_dir {datasets/QQP} --vocab bert --seq_len 128 --schedule_sampler lossaware --notes qqp

python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 40000 --save_interval 2000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --microbatch 64 --dataset qg --data_dir {datasets/QG} --vocab bert --seq_len 128 --schedule_sampler lossaware --notes qg

python -m torch.distributed.launch --nproc_per_node=7 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 140000 --save_interval 20000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --microbatch 64 --dataset dialogue --data_dir {datasets/Conversation} --vocab bert --seq_len 128 --schedule_sampler lossaware --notes dialogue

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 80000 --save_interval 20000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --microbatch 64 --dataset dialogue --data_dir {datasets/TS} --vocab bert --seq_len 128 --schedule_sampler lossaware --notes ts
```
Empirically, larger batchsize (larger `microbatch` here) can achieve higher BLEU score (without MBR). If you want to sync training loss to wandb, please customize your wandb setting in `train.py` (add your own API KEY).

## DiffuSeq Decoding
You need to modify the path to ```model_dir```, which is obtained in the training stage.
```bash
cd scripts
bash run_decode.sh
```
To reproduce the results of Table 1 in our paper, we suggest the size of MBR candidate set to be 10 (run 10 times using different seeds). Empirically, larger size can achieve higher BLEU score. For diversity metrics, the size of MBR candidate set is 3 when computing.

## Evaluation & MBR
You need to specify the folder of decoded texts. This folder should contain the decoded files from the same model but sampling with different random seeds. If ```mbr``` is not attached, we will compute the diversity score from the files in the folder, otherwise we will do MBR decoding:
```bash
cd scripts
python eval_seq2seq.py --folder ../{your-path-to-outputs} --mbr
```
Note: if you want to use this evaluation script for output files from other models, please make sure the same line from these output files refers to the same piece of data. Otherwise the diversity score could be incorrect.

## Update
- Update 22 May 2023: We prepare the checkpoint and sampling results for remaining tasks in this [link](https://drive.google.com/drive/folders/1lHPp-T-ytp-YVptiokeYK-Lth48EGQ12?usp=sharing).
- Update 28 Nov 2022: We prepare the checkpoint and sampling results of 10 seeds for QQP dataset in this [link](https://drive.google.com/drive/folders/1vnhJIUqPQva_x_sH2h5a0moCc1NYmEpr?usp=sharing).
- Update 14 Feb 2023: We update the evaluation scripts and camera ready version of the paper.

Welcome to discuss if you have any questions.

## Citation
Please add the citation if our paper or code helps you.

```
@inproceedings{gong2022diffuseq,
  author = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  booktitle = {International Conference on Learning Representations, ICLR},
  title = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  year = 2023
}
```
<p align = "center">
<img src="img/DiffuSeq_poster.png" width="100%" alt="" align=center />
</p>
<p align = "center">
DiffuSeq poster for ICLR 2023.
</p>