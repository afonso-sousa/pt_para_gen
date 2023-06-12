#!/bin/bash

model_name="output/assin2_sentence-transformers-paraphrase-multilingual-mpnet-base-v2-2023-04-19_14-00-23/1020"

CUDA_VISIBLE_DEVICES=1 python sts/test.py \
    --model_name $model_name