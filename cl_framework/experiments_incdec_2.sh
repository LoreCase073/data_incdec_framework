#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test/comparison_model_params/movinetA2/normal_sampler/lr4/patience10/ --data_path ./Kinetics --dataset kinetics --n_task 6 --n_class_first_task 5 --approach incdec --baseline --epochs 100 --nw 4 --seed 0 --backbone movinetA2 --batch_size 2 --scheduler_type reduce_plateau --plateau_check map --patience 10 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 --weight_decay 5e-4 --device 0 --sampler balanced --accumulation --n_accumulation 4 --stop_first_task 