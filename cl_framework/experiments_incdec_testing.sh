#!/bin/bash

python -u ./data_incdec_framework/cl_framework/main.py -op random_tries --data_path ./Kinetics --dataset kinetics --n_task 6 --n_class_first_task 5 --approach incdec --baseline --behaviors_check --epochs 1 --nw 4 --seed 0 --backbone movinetA0 --batch_size 4 --scheduler_type reduce_plateau --plateau_check map --patience 10 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 --weight_decay 5e-4 --device 1 --sampler imbalance_sampler --accumulation --n_accumulation 4 