#!/bin/bash


python -u ./data_incdec_framework/cl_framework/main.py -op first_test/no_class_weights_resnet/ --data_path ./Kinetics --dataset kinetics --n_task 2 --n_class_first_task 5 --approach incdec --baseline --epochs 80 --nw 4 --seed 0 --backbone 3dresnet18 --batch_size 4 --lr_first_task 1e-4 --device 1 