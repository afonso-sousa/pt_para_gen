#!/bin/bash

model_name="output/os-sts-subset_sentence-transformers-paraphrase-multilingual-mpnet-base-v2-2023-04-20_13-25-03/1020"

CUDA_VISIBLE_DEVICES=1 python sts/test.py \
    --model_name $model_name