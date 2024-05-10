#!/bin/bash
for i in 0
do
    python -u ./cl_framework/main.py -op /home/lorec/Documents/experiment_vzc/data_output/seed_$i/ \
        --approach incdec --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --restore_initial_parameters no \
        --fd_lamb 1 \
        --stop_first_task no \
        --epochs 2 --batch_size 2 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --multilabel yes --dataset vzctest --data_path /home/lorec/Documents/experiment_vzc/vzc_data_example \
        --subcategories_csv_path /home/lorec/Documents/experiment_vzc/vzc_data_example/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 1 --initial_split 2 --sampler imbalance_sampler \
        --subcategories_check yes --backbone movinetA0 --pretrained_path /data/cleaned_checkpoint_sgd.pt
done
