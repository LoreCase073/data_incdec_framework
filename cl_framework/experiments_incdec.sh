#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test --data_path ./Kinetics --dataset kinetics --n_task 3 --n_class_first_task 5 --approach incdec --epochs 10 --nw 4 --seed 0 --backbone 3dresnet18 --batch_size 4
