# European Portuguese Paraphrastic Dataset with Machine Translation

This repo contains the code for the paper European Portuguese Paraphrastic Dataset with Machine Translation, by Afonso Sousa & Henrique Lopes Cardoso.

We describe a new linguistic resource for the generation of paraphrases in Portuguese. This dataset comprises more than one million Portuguese-Portuguese sentential paraphrase pairs. We generated the pairs automatically by using neural machine translation to translate the non-Portuguese side of a large parallel corpus. We hope this new corpus can be a valuable resource for paraphrase generation and provide a rich semantic knowledge source to improve downstream natural language understanding tasks. To show the quality and utility of such a dataset, we use it to train paraphrastic sentence embeddings that outperform all other systems on ASSIN2 semantic textual similarity competition, in addition to showing how it can be used for paraphrase generation.

## Installation

First, create a fresh conda environment and install the dependencies:
```
conda create -n [ENV_NAME] python=3.9
pip install -r requirements.txt
```

## Build Dataset

**a. Create translated data.**

Run script [translate.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/translate.sh):
```shell
sh ./scripts/translate.sh
```
This script translates a subset of 2M entries. You might want to change the script to translate more/less entries.

**b. Filter data.**

Run script [filter.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/filter.sh):
```shell
sh ./scripts/filter.sh
```

**c. (optional) Make splits.**
Only the two steps above are required to build the dataset, but if you want to train semantic embeddings or a paraphrase generator, you should make splits.
Run script [make_splits.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/make_splits.sh):
```shell
sh ./scripts/make_splits.sh
```
This script randomly samples 300k entries and splits it in train/val/test sets of 80/10/10%, respectively. You might want to change the script to make splits with more/less entries.

**d. (optional). EDA.**
You can run a notebook-like file (with VSCode annotations) to get some data analysis.
```shell
python ./processed-data/eda.py
```

## Semantic Embeddings

### Train
Run script [sts/train.py](https://github.com/afonso-sousa/pt_para_gen/blob/main/sts/train.py):
```shell
python ./sts/train.py
```

### Test
Run script [sts/test.py](https://github.com/afonso-sousa/pt_para_gen/blob/main/sts/test.py):
```shell
python ./sts/test.py
```

## Paraphrase Generation
We train two models (M2M100 and mBart). Run the respective scripts. For example, to train and evaluate a mBart model, run the following scripts:

### Train
Run [scripts/train_mbart.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/train_mbart.py):
```shell
sh ./scripts/train_mbart.sh
```

### Test
For testing, we first generate the predictions and store them and in a different script compute the evaluate metrics.

**a. Generate predictions.**
Run script [scripts/gen_mbart.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/gen_mbart.py):
```shell
sh ./scripts/gen_mbart.sh
```

**b. Evaluate predictions.**
Run script [scripts/evaluate_mbart.sh](https://github.com/afonso-sousa/pt_para_gen/blob/main/scripts/evaluate_mbart.py):
```shell
sh ./scripts/evaluate_mbart.sh
```