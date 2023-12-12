#!/bin/bash
for i in 0 1 2 
do
    python -u ./data_incdec_framework/cl_framework/main.py -op runs_trainings/baseline_no_accumulation/normal/seed_$i/ --data_path ./Kinetics --dataset kinetics --n_task 6 --n_class_first_task 5 --approach incdec --baseline --epochs 100 --nw 4 --seed $i --backbone movinetA0 --batch_size 4 --scheduler_type reduce_plateau --plateau_check map --patience 10 --lr_first_task 1e-3 --head_lr 1e-3 --backbone_lr 1e-3 --weight_decay 5e-4 --device 0 --sampler balanced --behaviors_check
done