#!/usr/bin/env bash

model=UniCRS
dataset=redial

cd model/${model} || exit
cd data/${dataset}_rec || exit
python prepare.py

cd ../..

save_dir_prefix=../../checkpoint

prompt_encoder_root_dir=${save_dir_prefix}/${model}/pre/${dataset}
prompt_encoder_model_name=${dataset}-pre_ctx-128_resp-64_ent-43_lr-5e-4

output_root_dir=${save_dir_prefix}/${model}/rec/${dataset}
model_name=${dataset}-rec_ctx-128_ent-43_lr-1e-4

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_rec.py --kg ${dataset} --dataset ${dataset}_rec --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --prompt_encoder ${prompt_encoder_root_dir}/${prompt_encoder_model_name}/best --num_train_epochs 5 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_length 128 --resp_max_length 64 --entity_max_length 43 --learning_rate 1e-4 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}