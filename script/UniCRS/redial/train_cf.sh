#!/usr/bin/env bash

model=UniCRS
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

prompt_encoder_root_dir=${save_dir_prefix}/${model}/rec/${dataset}
prompt_encoder_model_name=${dataset}-rec_ctx-128_ent-43_lr-1e-4

output_root_dir=${save_dir_prefix}/${model}/cf/${dataset}
model_name=${dataset}_epoch-5_it-5_lr-0.5_l2-0.1_schema-0.9_flow_bs-200_beam-5_unicrs-rec-best_rec-1_lr-5e-5_l2-0.01_aug-pre

CUDA_VISIBLE_DEVICES=$1 accelerate launch train_cf.py --num_epochs 5 --strategy pre --kg_dataset ${dataset}-meta --flow_dataset ${dataset}_flow --split train --max_length 20 --pretrain_dir ${pretrain_root_dir}/${pretrain_model_name}/final --meta_path_model ${meta_path_root_dir}/${meta_path_model_name}/final/meta_path --use_meta_path --batch_size 200 --num_beams 5 --delta_iterations 5 --delta_lr 0.5 --delta_l2 0.1 --delta_l2_ratio 0.9 --crs_pre_dataset ${dataset}_pre --crs_rec_dataset ${dataset}_rec --crs_kg_dataset ${dataset} --tokenizer microsoft/DialoGPT-small --crs_model microsoft/DialoGPT-small --text_tokenizer roberta-base --crs_text_encoder roberta-base --crs_prompt_encoder ${prompt_encoder_root_dir}/${prompt_encoder_model_name}/best --crs_context_max_length 128 --crs_resp_max_length 64 --crs_entity_max_length 43 --crs_batch_size 64 --crs_pre_iterations 0 --crs_pre_lr 5e-5 --crs_rec_iterations 1 --crs_rec_lr 5e-5 --crs_l2 0.01 --output_dir ${output_root_dir}/${model_name} --fp16 --use_wandb --project CFCRS_${model} --name ${model_name}