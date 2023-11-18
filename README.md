# Depression Detection with Reddit and Self-training
**In Development** <br>
This repository contains the implementation code for paper: <br>
__Rethinking the Value of Labels for Improving Class-Imbalanced Learning__ <br>
[Dean Ninalga](justin.ninalga@mail.utoronto.ca) <br>
_Proceedings of the Third Workshop on Language Technology for Equality, Diversity and Inclusion_ <br>
as part of _The 14th International Conference on Recent Advances in Natural Language Processing (RANLP), 2023_ <br>
[[Paper](https://aclanthology.org/2023.ltedi-1.29/)]

If you find this code or idea useful, please consider citing our work:
```bib
@inproceedings{ninalga-2023-cordyceps-lt,
    title = "Cordyceps@{LT}-{EDI} : Depression Detection with {R}eddit and Self-training",
    author = "Ninalga, Dean",
    editor = "Chakravarthi, Bharathi R.  and
      Bharathi, B.  and
      Griffith, Joephine  and
      Bali, Kalika  and
      Buitelaar, Paul",
    booktitle = "Proceedings of the Third Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ltedi-1.29",
    pages = "192--197",
}
```


# Overview
In this work, we show that domain relevant __semi-supervised learning__ (using unlabeled data) through __self-training__ (student learns from labels produced by teacher) yields highly competitive results (3rd overall in the [DepSign-2023](https://aclanthology.org/2023.ltedi-1.4.pdf) shared task) in the depression detection setting. <br>
In our analysis we highlighted ADHD-focused forums as a major source of (non-diagnostic) severe depression classifications suggesting some level of overlapping language or verbal processes shared between persons with ADHD and/or depression.
![Screenshot](bargraph.png)

## Key Requirements
###  Installation
- CUDA 11.8 compatible GPU
- PyTorch >= 2.1.0
- Transformers >= 4.35.1
- Numpy, Pandas
```bash
conda create -n langP python=3.9
conda activate langP 
# Optional: Install CUDA via conda for a smoother installation experience,
# but you may need to manually set the Anaconda path variables.
# conda install cuda -c nvidia/label/cuda-11.8.0
pip install torch==2.1.0
pip install transformers==4.35.1
pip install -q wget # for downloading datasets
pip install -q loguru # logging
```

### Accessing Gated Mental-Health models
Many huggingface language models for mental health such as 
[MentalBERT](https://huggingface.co/mental/mental-bert-base-uncased) and [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base)
require special access privileges to help prevent the misuse of these models. 
To access these models you have to read and accept the conditions outlined on the huggingface web page (e.g. [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base)).
Once you have been granted access to the model, you then have to connect your environment to your huggingface account, which can be performed using the following steps. <br>

Download the huggingface cli package:
```commandline
pip install --upgrade huggingface_hub
```
Then sign-in using you api-key using the following:
```commandline
huggingface-cli login
```
Follow the link and instructions provided by the output of the above command to sign-in. <br>
You should now be able to download the models that have been granted access to.


## Code Overview
#### Main Files
- [`train_semi.py`](train_semi.py): train model with generated labels of unlabeled data
- [`train.py`](train.py): train model with shared task data
- [`gen_pseudolabels.py`](gen_pseudolabels.py): generate labels of unlabeled data using trained model

#### Main Arguments
- `--dataset`: name of chosen unlabeled dataset, on RMHD / RMHD-small
- `--model_name`: name of backbone huggingface model (e.g. 'bert-base-uncased', 'roberta-base', etc.)

## Getting Started
### Finetuning
Train on DepSign2023 using MentalRoBERTa
```commandline
python depr-selftrain/train.py --model_name "mental/mental-roberta-base" 
```

### Semi-Supervised Learning
#### (Optional) Smaller unlabeled data sources
We use the [Reddit Mental Health Dataset](https://zenodo.org/records/3941387) (RMHD) to source unlabeled data which may be memory prohibitive.
Use `--dataset rmhd-small` to use a subset of the RMHD using only data from depression focused forums / subreddits.

#### Semi-supervised learning with pseudo-labeling
To perform pseudo-labeling (self-training), first a base classifier is trained on original dataset.
```commandline
python depr-selftrain/train.py --model_name "mental/mental-roberta-base" 
```
With the trained base classifier, pseudo-labels can be generated using
```commandline
python depr-selftrain/gen_pseudolabels.py --dataset rmhd-small \
  --trained_model "output/base/model-best.pt" --model_name "mental/mental-roberta-base"
```
To train a new models with the newly generated pseudolabels
```commandline
python depr-selftrain/train_semi.py --dataset rmhd --pseudolabels rmhd-small-predictions.npy \
  --model_name "mental/mental-roberta-base" --num_epochs 2
```


# Acknowledgements
This code is partly based on the open-source implementations from the following sources:
[imbalanced-semi-self](https://github.com/YyzHarry/imbalanced-semi-self/tree/master) <br>
For this work we rely on the [Reddit Mental Health Dataset](https://zenodo.org/records/3941387) for unlabeld data. Check out [the original paper on JMIR](https://www.jmir.org/2020/10/e22635/).
