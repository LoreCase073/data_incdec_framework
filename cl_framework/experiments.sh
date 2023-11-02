#!/bin/bash

filename=$0
filename=${filename%.*}
mkdir -p $1/$filename
 



python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 6 --n_class_first_task 50 --approach efc --epochs 100 --nw 12 --seed 0 

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 11 --n_class_first_task 50 --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 21 --n_class_first_task 40 --approach efc --epochs 100 --nw 12 --seed 0


python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 6  --n_class_first_task 100 --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 11  --n_class_first_task 100 --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 21 --n_class_first_task 100 --approach efc --epochs 100 --nw 12 --seed 0



python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 11 --n_class_first_task 50 --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 21 --n_class_first_task 40 --approach efc --epochs 100 --nw 12 --seed 0