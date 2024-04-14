#!/bin/bash

python_exec='/local/xiaowang/miniconda3/envs/food_ingre/bin/python'
script_path='/local/xiaowang/food_ingredient/approach2/title2ingre/run_text_classification.py'

export CUDA_VISIBLE_DEVICES="4"
export WANDB_PROJECT="ingre_approach2"
export WANDB_USERNAME="loss4wang"
export WANDB_API_KEY="81a3c9c3fa59f92cbf80af38f03905c8ea01a3fc"
export WANDB_NAME="deberta_instruction_lr5" #! need change each run


$python_exec "$script_path" \
    --model_name_or_path "microsoft/deberta-v3-base" \
    --trust_remote_code "True" \
    --train_file "/local/xiaowang/food_ingredient/Dataset_440_labels/train_set.json" \
    --validation_file "/local/xiaowang/food_ingredient/Dataset_440_labels/val_set.json" \
    --test_file "/local/xiaowang/food_ingredient/Dataset_440_labels/test_set.json" \
    --is_single_classification "False"\
    --label_column_name "cleaned_ingredients" \
    --text_column_names "title,generated_intro" \
    --text_column_delimiter "[SEP]" \
    --max_seq_length "512" \
    --pad_to_max_length "False" \
    --shuffle_train_dataset "False" \
    --seed "42" \
    --per_device_train_batch_size "32" \
    --optim "adamw_torch" \
    --evaluation_strategy "epoch" \
    --early_stopping_patience "3" \
    --save_strategy "epoch" \
    --logging_strategy "epoch" \
    --num_train_epochs "200.0" \
    --learning_rate "7e-5" \
    --load_best_model_at_end "True" \
    --do_train "True" \
    --do_eval "True" \
    --do_predict "True" \
    --output_dir "/local/xiaowang/food_ingredient/saved_models/approach2/text_classification_generated_lr5" \
    --overwrite_output_dir "True" \
    --report_to "wandb"

