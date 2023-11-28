#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test/movinet_6tasks_with_weighted_sampler/ --data_path ./Kinetics --dataset kinetics --n_task 6 --n_class_first_task 5 --approach incdec --baseline --epochs 80 --nw 4 --seed 0 --backbone movinet --batch_size 3 --scheduler_type multi_step --lr_first_task 1e-4 --head_lr 1e-5 --backbone_lr 1e-5 --early_stopping_val 40 --device 1 --sampler imbalance_sampler