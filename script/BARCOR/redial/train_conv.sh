#!/usr/bin/env bash

model=BARCOR
dataset=redial

cd model/${model} || exit

cd data/${dataset}_conv || exit
python prepare.py

cd ../${dataset}-cf_conv || exit
python prepare.py

cd ../..

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/conv/${dataset}
model_name=${dataset}+cf_ctx-160_resp-128_bs-64_lr-1e-4_epoch-15

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_conv.py --kg_dataset ${dataset} --dataset ${dataset}_conv --other_dataset ${dataset}-cf_conv --context_max_length 160 --resp_max_length 128 --tokenizer facebook/bart-base --conv_model facebook/bart-base --num_train_epochs 15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 1e-4 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}