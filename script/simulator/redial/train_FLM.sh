#!/usr/bin/env bash

model=FLM
dataset=redial

cd model/simulator || exit

cd data/${dataset}_flow || exit
python prepare.py

cd ../${dataset}_schema-5_walk-10000_flow || exit
python prepare.py
cd ../../

save_dir_prefix=../../checkpoint

output_root_dir=${save_dir_prefix}/simulator/${model}/${dataset}
model_name=${dataset}+schema-5_walk-10000_flow_enc-12_dec-12_epoch-15_lr-5e-5

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_flow.py --dataset ${dataset}_flow --other_dataset ${dataset}_schema-5_walk-10000_flow --kg_dataset ${dataset}-meta --use_meta_path --max_length 20 --encoder_layers 12 --decoder_layers 12 --num_train_epochs 15 --per_device_train_batch_size 512 --per_device_eval_batch_size 256 --learning_rate 5e-5 --fp16 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}
