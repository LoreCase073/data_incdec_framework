#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test/with_class_weights/lower_res --data_path ./Kinetics --dataset kinetics --n_task 3 --n_class_first_task 5 --approach incdec --epochs 100 --nw 4 --seed 0 --backbone movinet --batch_size 2
