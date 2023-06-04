#!/usr/bin/env bash

model=BARCOR
dataset=redial

cd model/${model} || exit

cd data/${dataset}_flow || exit
python prepare.py

cd ../..

save_dir_prefix=../../checkpoint

pretrain_root_dir=${save_dir_prefix}/simulator/FLM/${dataset}
pretrain_model_name=${dataset}+schema-5_walk-10000_flow_enc-12_dec-12_epoch-15_lr-5e-5

meta_path_root_dir=${save_dir_prefix}/simulator/schema/${dataset}
meta_path_model_name=${dataset}+schema-5_walk-10000_flow_enc-12_dec-12_epoch-20_lr-5e-5_final_bs-256_lr-1e-4

crs_model_root_dir=${save_dir_prefix}/${model}/rec/${dataset}
crs_model_name=${dataset}_ctx-160_bs-128_lr-1e-4

output_root_dir=${save_dir_prefix}/${model}/cf/${dataset}
model_name=${dataset}_epoch-5_it-5_lr-0.5_l2-0.001_decay-0.9_flow_bs-150_beam-5_bart-final_it-1_lr-5e-5_l2-0.01_aug-pre

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_cf.py --num_epochs 5 --strategy pre --kg_dataset ${dataset}-meta --flow_dataset ${dataset}_flow --split train --max_length 20 --pretrain_dir ${pretrain_root_dir}/${pretrain_model_name}/final --meta_path_model ${meta_path_root_dir}/${meta_path_model_name}/final/meta_path --use_meta_path --user_policy cf --flow_policy model --batch_size 150 --num_beams 5 --delta_iterations 5 --delta_lr 0.5 --delta_l2 0.001 --delta_l2_ratio 0.9 --crs_dataset ${dataset}_rec --crs_kg_dataset ${dataset} --tokenizer facebook/bart-base --crs_model ${crs_model_root_dir}/${crs_model_name}/final --crs_context_max_length 160 --crs_batch_size 64 --crs_iterations 1 --crs_lr 5e-5 --crs_l2 0.01 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}