#!/bin/bash
for i in 0
do
    python -u ./data_incdec_framework/cl_framework/main.py -op random_tries/seed_$i/ \
        --approach incdec_lwf --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --restore_initial_parameters no \
        --fd_lamb 1 \
        --stop_first_task no \
        --epochs 1 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 1 \
        --criterion_type multilabel --multilabel no --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path ./Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_class_first_task 5 --n_task 6 --initial_split 2 --valid_size 0.0 --sampler imbalance_sampler \
        --subcategories_check yes --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

