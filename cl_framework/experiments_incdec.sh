#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test/movinet_6tasks/ --data_path ./Kinetics --dataset kinetics --n_task 6 --n_class_first_task 5 --approach incdec --baseline --epochs 80 --nw 4 --seed 0 --backbone movinet --batch_size 3s --lr_first_task 5e-4 --device 1