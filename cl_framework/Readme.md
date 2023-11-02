
This repository contains all code needed to reproduce the experimental results in the main paper.

# Setting up the Conda environment

To run the code you must create an Anaconda environment from the `environment.yml` file and activate it:

```
conda env create -n il_maru -f environment.yml 
conda activate il_maru
```

# Running the code for CIFAR-100 experiments

The default hyperparameters are the ones used to compute the Table 1 in the main paper.

1. 6 Task

```
python -u   main.py -op ./cifar100_6task --dataset cifar100 --n_task 6 --n_class_first_task 50 --approach lwf --epochs 100 --nw 12 --seed 0 

```

2. 11 Task

```
python -u   main.py -op ./cifar100_11task --dataset cifar100 --n_task 11 --n_class_first_task 50 --approach lwf --epochs 100 --nw 12 --seed 0 

```

3. 21 Task

```
python -u   main.py -op ./cifar100_21task --dataset cifar100 --n_task 21 --n_class_first_task 40 --approach lwf --epochs 100 --nw 12 --seed 0 

```

# Running the Tiny-Imagenet and ImageNet-Subset experiments

The commands are similar, the only difference is that  the  data-folder "cl_data" where both the datasets are downloaded should be specified.

Here we provide the 6 task scenario and the 11 task scenario for TinyImageNet and ImageNet-Subset.

```
python -u   main.py -op ./tinyimagenet_6task  --dataset tiny-imagenet  --n_task 6  --n_class_first_task 100 --approach lwf  --nw 12 --seed 0 --data_path ./cl_data
```

```
python -u   main.py -op ./imagenetsubset_11task --dataset imagenet-subset  --n_task 11  --n_class_first_task 50 --approach lwf   --nw 12  --seed 0 --data_path ./cl_data
```

In the bash file "experiments.sh" all the experiments for all the scenarios can be run.

# Main command-line arguments

* `-op`: folder path where results are stored
* `--nw`: number of workers for data loaders
* `--epochs`: total number of epochs for the incremental steps (default=100)
* `--seed`: random seed (default=0)
* `--n_task`:  number of task, included the  first task
* `--n_class_first_task`:  number of classes in the first task
* `--data_path`: data folder where imagenet subset and tiny-imagenet are stored.

# Analyzing the results

The results are stored in path specified by `-op`.  A file
`summary.csv` with the command line arguments and the performance will
be generated. The last two columns `Last_avg_tag_acc`,`
Last_avg_perstep_tag_acc` represent, respectively, the average accuracy
across the task (Right Formula in Equation 15 in the the main paper)
and the per step incremental accuracy (Left Formula in Equation 15 in
the main paper).

 
