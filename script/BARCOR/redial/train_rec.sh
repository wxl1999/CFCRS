#!/usr/bin/env bash

model=BARCOR
dataset=redial

cd model/${model} || exit
cd data/${dataset}_rec || exit
python prepare.py

cd ../..

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/rec/${dataset}
model_name=${dataset}_ctx-160_bs-128_lr-1e-4

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_rec.py --kg_dataset ${dataset} --dataset ${dataset}_rec --context_max_length 160 --tokenizer facebook/bart-base --model facebook/bart-base --num_train_epochs 5 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --learning_rate 1e-4 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}