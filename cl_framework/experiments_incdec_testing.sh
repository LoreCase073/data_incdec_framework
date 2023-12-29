#!/bin/bash

python -u ./data_incdec_framework/cl_framework/main.py -op random_tries/to_remove2/ \
    --approach incdec --pipeline baseline --n_accumulation 4 --seed 0 --nw 4 \
    --freeze_bn no --early_stopping_val 10 --weight_decay 5e-4 --stop_first_task no \
    --epochs 20 --batch_size 4 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
    --scheduler_type reduce_plateau --plateau_check map --patience 10 --device 0 \
    --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
    --behaviors_to_remove_csv_path ./Kinetics/Info/behaviors_to_remove.csv\
    --n_class_first_task 5 --n_task 3 --initial_split 2 --valid_size 0.0 --sampler balanced \
    --behaviors_check no --backbone movinetA0
