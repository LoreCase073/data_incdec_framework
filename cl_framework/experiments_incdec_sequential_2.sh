#!/bin/bash
for i in 0 1 2 
do
    python -u ./data_incdec_framework/cl_framework/main.py -op runs_trainings/fd/baseline/lambda_01/seed_$i/ \
        --approach incdec_fd --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --restore_initial_parameters no \
        --lwf_lamb 1 --lwf_T 1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type reduce_plateau --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --behaviors_csv_path None --behaviors_randomize yes \
        --n_class_first_task 5 --n_task 6 --initial_split 2 --valid_size 0.0 --sampler imbalance_sampler \
        --behaviors_check yes --backbone movinetA0
done