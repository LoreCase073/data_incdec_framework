#!/bin/bash
for i in 0 1 2 
do
    python -u ./data_incdec_framework/cl_framework/main.py -op runs_trainings/multilabel/normal/seed_$i/ --approach incdec --pipeline baseline --n_accumulation 4 --seed $i --nw 4 --freeze_bn no --early_stopping_val 10 --weight_decay 5e-4 --stop_first_task no --epochs 100 --batch_size 4 --lr_first_task 1e-3 --head_lr 1e-3 --backbone_lr 1e-3 --scheduler_type reduce_plateau --plateau_check map --patience 10 --device 0 --criterion_type multilabel --dataset kinetics --data_path ./Kinetics --n_class_first_task 5 --n_task 6 --initial_split 2 --valid_size 0.0 --sampler balanced --behaviors_check yes --backbone movinetA0
done