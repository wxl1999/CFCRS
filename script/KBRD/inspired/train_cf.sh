#!/usr/bin/env bash

model=KBRD
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
crs_model_name=${dataset}_hs-128_epoch-5_batch-64_lr-1e-4

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/cf/${dataset}
model_name=${dataset}_epoch-10_it-10_lr-0.5_l2-0.1_decay-0.95_flow_bs-50_beam-20_kbrd-final_it-1_lr-1e-4_l2-0.01_aug

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_cf.py --num_epochs 10 --kg_dataset ${dataset}-meta --flow_dataset ${dataset}_flow --split train --max_length 20 --pretrain_dir ${pretrain_root_dir}/${pretrain_model_name}/final --meta_path_model ${meta_path_root_dir}/${meta_path_model_name}/final/meta_path --batch_size 50 --num_beams 20 --aug_num 1 --delta_iterations 10 --delta_lr 0.5 --delta_l2 0.1 --delta_l2_ratio 0.95 --crs_dataset ${dataset}_rec --crs_kg_dataset ${dataset} --crs_model ${crs_model_root_dir}/${crs_model_name}/final --crs_max_length 64 --crs_hidden_size 128 --crs_batch_size 64 --crs_iterations 1 --crs_lr 1e-4 --crs_l2 0.01 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}