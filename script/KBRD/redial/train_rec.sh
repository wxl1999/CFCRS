#!/usr/bin/env bash

model=KBRD
dataset=redial

cd model/${model} || exit
cd data/${dataset}_rec || exit
python prepare.py

cd ../..

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/rec/${dataset}
model_name=${dataset}_hs-128_epoch-5_batch-64_lr-5e-4

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_rec.py --kg_dataset ${dataset} --dataset ${dataset}_rec --entity_max_length 64 --hidden_size 128 --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate 5e-4 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}