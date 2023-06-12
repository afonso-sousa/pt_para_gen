#!/bin/bash

model_name="output/tapaco-sts-subset_sentence-transformers-paraphrase-multilingual-mpnet-base-v2-2023-04-20_13-13-45/1020"

CUDA_VISIBLE_DEVICES=1 python sts/test.py \
    --model_name $model_name