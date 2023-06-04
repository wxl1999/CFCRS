#!/usr/bin/env bash

model=UniCRS
dataset=inspired

cd model/${model} || exit
cd data/${dataset}_pre || exit
python prepare.py

cd ../..

save_dir_prefix=../../checkpoint

output_root_dir=${save_dir_prefix}/${model}/pre/${dataset}
model_name=${dataset}-pre_ctx-128_resp-128_ent-31_lr-1e-3

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_pre.py --kg ${dataset} --dataset ${dataset}_pre --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --max_length 128 --resp_max_length 128 --entity_max_length 31 --learning_rate 1e-3 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}