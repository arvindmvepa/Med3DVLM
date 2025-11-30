#!/bin/bash

output_dir=./output/finetuned-model-new-dataset-v11-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json
test_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_updated_v11_seed0_multitask_fixed.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa.py \
    --model_name_or_path "$output_dir"/hf \
    --vqa_data_test_path $test_path \
    --max_length 512 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir $output_dir/eval_vqa \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa \
--gt_file $test_path \


output_dir=./output/finetuned-model-new-dataset-met-v11-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json
test_path=/local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_test_updated_v11_seed0_multitask_fixed.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa.py \
    --model_name_or_path "$output_dir"/hf \
    --vqa_data_test_path $test_path \
    --max_length 512 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir $output_dir/eval_vqa \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa \
--gt_file $test_path \


output_dir=./output/finetuned-model-new-dataset-goat-v11-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_goat_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_goat_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json
test_path=/local2/amvepa91/MedTrinity-25M/brats_goat_3d_vqa_subjTrue_test_updated_v11_seed0_multitask_fixed.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa.py \
    --model_name_or_path "$output_dir"/hf \
    --vqa_data_test_path $test_path \
    --max_length 512 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir $output_dir/eval_vqa \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python src/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa \
--gt_file $test_path \