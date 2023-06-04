#!/usr/bin/env bash

model=BARCOR
dataset=inspired

cd model/${model} || exit

cd data/${dataset}_flow || exit
python prepare.py

cd ../..

pretrain_root_dir=/mnt/wangxiaolei/CFCRS/simulator/FLM/${dataset}
pretrain_model_name=${dataset}+schema-2_walk-10000_flow_enc-12_dec-12_epoch-20_lr-5e-5
#pretrain_root_dir=/mnt/wangxiaolei/crs/dialog_sparse
#pretrain_model_name=enc-12_dec-12_inspired+schema-2_walk-10000_flow_5e-5_20

meta_path_root_dir=/mnt/wangxiaolei/CFCRS/simulator/schema/${dataset}
meta_path_model_name=${dataset}+schema-2_walk-10000_flow_enc-12_dec-12_epoch-20_lr-5e-5_final_bs-256_lr-1e-4
#meta_path_root_dir=/mnt/wangxiaolei/crs/dialog_sparse
#meta_path_model_name=enc-12_dec-12_inspired+schema-2_walk-10000_flow_5e-5_20_final_bs-256_lr-1e-4

crs_model_root_dir=/mnt/wangxiaolei/CFCRS/${model}/rec/${dataset}
crs_model_name=${dataset}_ctx-160_bs-128_lr-1e-4

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/cf/${dataset}
model_name=${dataset}_epoch-5_it-10_lr-0.5_l2-0.001_decay-0.9_flow_bs-80_beam-20_cf_bart-final_it-1_lr-5e-5_l2-0.01_aug-mix

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_cf.py --num_epochs 5 --strategy mix --kg_dataset ${dataset}-meta --flow_dataset ${dataset}_flow --split train --max_length 16 --pretrain_dir ${pretrain_root_dir}/${pretrain_model_name}/final --meta_path_model ${meta_path_root_dir}/${meta_path_model_name}/final/meta_path --use_meta_path --batch_size 80 --num_beams 20 --delta_iterations 10 --delta_lr 0.5 --delta_l2 0.001 --delta_l2_ratio 0.9 --crs_dataset ${dataset}_rec --crs_kg_dataset ${dataset} --tokenizer facebook/bart-base --crs_model ${crs_model_root_dir}/${crs_model_name}/final --crs_context_max_length 160 --crs_batch_size 64 --crs_iterations 1 --crs_lr 5e-5 --crs_l2 0.01 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}