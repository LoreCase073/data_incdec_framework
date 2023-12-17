# data_incdec_framework

Repo for the Data Incremental/Decremental of the master thesis project.

# Setting up the environment

To run the code you must create an Anaconda environment from the `environment.yml` file and activate it:

```
conda env create -n data_incdec -f environment.yml 
conda activate data_incdec
```

# Running the code for Data Incremental/Decremental pipelines

To run the code you can use the command line interface with a prompt like the one:

```
python -u ./data_incdec_framework/cl_framework/main.py -op directory_out \
    --approach incdec --pipeline baseline --n_accumulation 4 --seed 0 --nw 4 \
    --freeze_bn no --early_stopping_val 10 --weight_decay 5e-4 --stop_first_task no \
    --epochs 100 --batch_size 4 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
    --scheduler_type reduce_plateau --plateau_check map --patience 10 --device 0 \
    --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
    --n_class_first_task 5 --n_task 6 --initial_split 2 --valid_size 0.0 --sampler imbalance_sampler \
    --behaviors_check yes --backbone movinetA0
```

For VZC case, the example of prompt it is:

```
python -u ./data_incdec_framework/cl_framework/main.py -op directory_out \
    --approach incdec --pipeline baseline --n_accumulation 4 --seed 0 --nw 4 \
    --freeze_bn no --early_stopping_val 10 --weight_decay 5e-4 --stop_first_task no \
    --epochs 100 --batch_size 4 --lr_first_task 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
    --scheduler_type reduce_plateau --plateau_check map --patience 10 --device 0 \
    --criterion_type multilabel --dataset vzc --data_path ./path_to_dataset \
    --n_class_first_task 3 --n_task 6 --initial_split 2 --valid_size 0.0 --sampler balanced \
    --behaviors_check no --backbone movinetA0
```

All the hyperparameters that matter for this case are listed below.

# Command-line arguments
* `-op`: folder path where results are stored
* `--approach`: Type of machine learning approach to be followed. Choices are: ["finetuning", "ewc","lwf","incdec"]. (default=incdec)
* `--pipeline`: Type of pipeline to be follower in the incdec case. For now only "baseline" is available. (default=baseline)
* `--n_accumulation`: To be used in case batch size is low and you want to do gradient accumulation. If 0, no gradient accumulation will be used, else it will accumulate for n batches. (default=0)
* `--seed`: random seed (default=0)
* `--freeze_bn`: If training need to be done with the backbone frozen. Choices: ['yes','no']. (default=no)
* `--early_stopping_val`: If need to do early stopping, right now not implemented. (default=1000)
* `--weight_decay`: (default=5e-4)
* `--stop_first_task`: To stop at first task, needed in debugging. Choices: ["yes","no"]. (default=no)
* `--epochs`: (default=100)
* `--batch_size`: (default=4)
* `--lr_first_task`: (default=1e-4)
* `--head_lr`: (default=1e-4)
* `--backbone_lr`: (default=1e-4)
* `--scheduler_type`: Choices: ["fixd", "multi_step","reduce_plateau"] (default=reduce_plateau)
* `--plateau_check`: Select the metric to be checked by reduce_plateau scheduler. Mean Average precision 'map' or classification loss 'class_loss. Choices: ["map", "class_loss"] (default=reduce_plateau)
* `--plateau_check`: Patience for the reduce_plateau scheduler. (default=10)
* `--device`: (default=0)
* `--criterion_type`: Select the type of loss to be used, for multiclass is cross_entropy, for multilabel BCE. Choices: ["multiclass", "multilabel"]. (default=multilabel)
* `--dataset`: dataset to use. Choices: ["cifar100","tiny-imagenet","imagenet-subset", "kinetics", "vzc"]. (default=kinetics)
* `--data_path`: path where dataset is saved. (default=./Kinetics)
* `--n_class_first_task`: if greater than -1 use a larger number of classes for the first class, n_task include this one. If incdec approach set it to the number of classes of all tasks. (default=5)
* `--n_task`: number of tasks (default=6)
* `--initial_split`: how to divide in the initial split the dataset. 2 will divide in 50%/50%. Other types of split will be implemented later. Choices: [2] (default=2)
* `--valid_size`: percentage of train for validation set, default not use validation set. Especially for incdec not implemented (default=0.0)
* `--sampler`: Select the type of sampler to ber used by dataloader. imbalance sampler is for class imbalance cases. balanced is the standard one. Choices: ["imbalance_sampler","balanced"]. (default=balanced)
* `--behaviors_check`: Ff we want to work with behaviors (subcategories). Choices=["yes", "no"] (default=yes)
* `--backbone`: Choices=["resnet18","3dresnet18","3dresnet10","movinetA0","movinetA1","movinetA2"] (default=movinetA0)
* `--firsttask_modelpath`: specify model path if start from a pre-trained model after first task (default=None) 


