#!/bin/bash

output_dir=./output/finetuned-model-new-dataset-v1-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_updated_v10_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_updated_v10_seed0_multitask_fixed.json
test_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_updated_v10_seed0_multitask_fixed.json

CUDA_VISIBLE_DEVICES=$1 deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name Med3DVLM-Qwen-2.5-7B-finetune \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --lora_enable True \
    --vision_select_layer -2 \
    --pretrain_vision_model ./output/DCFormer_SigLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter ./output/Med3DVLM-Qwen-2.5-7B-pretrain/mm_projector.safetensors \
    --vqa_data_train_path $train_path \
    --vqa_data_val_path $val_path \
    --vqa_data_test_path $test_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4

python src/utils/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --pretrain_vision_model ./output/DCFormer_SigLIP/pretrained_ViT.bin \
    --vision_tower "dcformer" \
    --model_with_lora "$output_dir"/model_with_lora.bin \
    --output_dir="$output_dir"/hf
