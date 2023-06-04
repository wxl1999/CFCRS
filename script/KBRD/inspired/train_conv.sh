#!/usr/bin/env bash

model=KBRD
dataset=inspired

cd model/${model} || exit

cd data/${dataset}_conv || exit
python prepare.py

cd ../${dataset}-cf_conv || exit
python prepare.py

cd ../..

rec_root_dir=/mnt/wangxiaolei/CFCRS/${model}/cf/${dataset}/
rec_model_name=${dataset}_epoch-10_it-10_lr-0.5_l2-0.1_decay-0.95_flow_bs-50_beam-20_kbrd-final_it-1_lr-1e-4_l2-0.01_aug

output_root_dir=/mnt/wangxiaolei/CFCRS/${model}/conv/${dataset}
model_name=${dataset}+cf_ctx-200_resp-128_ent-32_text-300_ent-128_enc-2_dec-2_head-2_bs-128_lr-5e-4_epoch-20

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_conv.py --kg_dataset ${dataset} --dataset ${dataset}_conv --other_dataset ${dataset}-cf_conv --context_max_length 200 --resp_max_length 128 --entity_max_length 32 --rec_model ${rec_root_dir}/${rec_model_name}/best --tokenizer facebook/bart-base --text_hidden_size 300 --entity_hidden_size 128 --encoder_layers 2 --decoder_layers 2 --attn_head 2 --num_train_epochs 20 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --learning_rate 5e-4 --output_dir ${output_root_dir}/${model_name} --use_wandb --project CFCRS_${model} --name ${model_name}