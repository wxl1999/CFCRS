#!/usr/bin/env bash

model=UniCRS
dataset=inspired

cd model/${model} || exit
cd data/${dataset}_rec || exit
python prepare.py

cd ../..

prompt_encoder_root_dir=/mnt/wangxiaolei/CFCRS/${model}/pre/${dataset}
prompt_encoder_model_name=${dataset}-pre_ctx-128_resp-128_ent-31_lr-1e-3

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/rec/${dataset}
model_name=${dataset}-rec_ctx-256_ent-31_lr-1e-4_no-sch

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_rec.py --kg ${dataset} --dataset ${dataset}_rec --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --prompt_encoder ${prompt_encoder_root_dir}/${prompt_encoder_model_name}/final --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --max_length 256 --resp_max_length 128 --entity_max_length 31 --learning_rate 1e-4 --no_lr_scheduler --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}