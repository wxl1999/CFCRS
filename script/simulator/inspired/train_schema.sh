#!/usr/bin/env bash

model=schema
dataset=inspired

cd model/simulator || exit

cd data/${dataset}_schema || exit
python prepare.py

cd ../${dataset}_schema-2_walk-10000_schema || exit
python prepare.py
cd ../../

FLM_root_dir=/mnt/wangxiaolei/CFCRS/simulator/FLM/${dataset}
FLM_model_name=${dataset}+schema-2_walk-10000_flow_enc-12_dec-12_epoch-20_lr-5e-5

output_root_dir=/mnt/wangxiaolei/CFCRS/simulator/${model}/${dataset}
model_name=${dataset}+schema-2_walk-10000_flow_enc-12_dec-12_epoch-20_lr-5e-5_final_bs-256_lr-1e-4

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_meta.py --kg_dataset ${dataset}-meta --dataset ${dataset}_schema --other_dataset ${dataset}_schema-2_walk-10000_schema --max_length 16 --pretrain_dir ${FLM_root_dir}/${FLM_model_name}/final --num_train_epochs 5 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --learning_rate 1e-4 --fp16 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}
