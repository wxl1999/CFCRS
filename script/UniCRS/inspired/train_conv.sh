#!/usr/bin/env bash

model=UniCRS
dataset=inspired

cd model/${model} || exit

cd data/${dataset}_conv || exit
python prepare.py

cd ../${dataset}-cf_conv || exit
python prepare.py

cd ../..

prompt_encoder_root_dir=/mnt/wangxiaolei/CFCRS/${model}/pre/${dataset}
prompt_encoder_model_name=${dataset}-pre_ctx-128_resp-128_ent-31_lr-1e-3

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/conv/${dataset}
model_name=${dataset}-conv+cf_pre-final_ctx-128_resp-128_ent-43_lr-5e-4

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_conv.py --kg_dataset ${dataset} --dataset ${dataset}_conv --other_dataset ${dataset}-cf_conv --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base --prompt_encoder ${prompt_encoder_root_dir}/${prompt_encoder_model_name}/final --num_train_epochs 15 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --entity_max_length 43 --learning_rate 5e-4 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}