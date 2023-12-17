import argparse
from ast import parse
import sys
    
def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--approach", type=str,default="incdec", choices=["finetuning", "ewc","lwf","incdec"], help="Type of machine learning approach to be followed.")
    parser.add_argument("--pipeline", type=str,default="baseline", choices=["baseline",], help="Type of pipeline to be follower in the incdec case.") 
    parser.add_argument("--n_accumulation", type=int, default=0, help="To be used in case batch size is low and you want to do gradient accumulation.")
    parser.add_argument("--outpath", "-op",default="./", type=str, help="Output directory where to save results.") 
    parser.add_argument("--seed", type=int, default=0, help="Seed to be initialized to.")
    parser.add_argument("--nw", type=int, default=4, help="num workers for data loader")
    parser.add_argument("--freeze_bn", type=str, default="no", choices=["yes", "no"], help="If training need to be done with the backbone frozen. Choices: ['yes','no']")
    parser.add_argument("--early_stopping_val", type=int, default=1000, help="If need to do early stopping, right now not implemented.")
    parser.add_argument("--weight_decay", default=5e-4, type=float)

    
    """
    EWC Hyperparams
    """
    parser.add_argument("--ewc_lamb", default=5000.0, type=float, help="")
    parser.add_argument("--ewc_alpha", default=0.5, type=float)
    """
    LWF hyperparams 
    """
    parser.add_argument("--lwf_lamb", default=10.0, type=float, help="")
    parser.add_argument("--lwf_T", default=2.0, type=float, help="")
 
    """
    Training hyperparams 
    """
    parser.add_argument("--stop_first_task", type=str, default="no", choices=["yes", "no"], help="Flag to stop at first task, needed in debugging.") 
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_first_task", type=float, default=1e-4, help="for tiny-imagenet and cifar100, for imagenet-subet default  are in IncrementalApproach.py")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="reduce_plateau", choices=["fixd", "multi_step","reduce_plateau"])
    parser.add_argument("--plateau_check", type=str, default="map", choices=["map", "class_loss"], help="Select the metric to be checked by reduce_plateau scheduler. Mean Average precision 'map' or classification loss 'class_loss'.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for the reduce_plateau scheduler.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--criterion_type", type=str, default="multilabel", choices=["multiclass", "multilabel"], help="Select the type of loss to be used, for multiclass is cross_entropy, for multilabel BCE.")
    
    "Dataset Settings"
    parser.add_argument("--dataset", type=str, default="kinetics", choices=["cifar100","tiny-imagenet","imagenet-subset", "kinetics", "vzc"], help="dataset to use") 
    parser.add_argument("--data_path",type=str, default="/Kinetics",help="path where dataset is saved")
    parser.add_argument("--n_class_first_task", type=int, default=5, help="if greater than -1 use a larger number of classes for the first class, n_task include this one. If incdec approach set it to the number of classes of all tasks.")
    parser.add_argument("--n_task", type=int, default=6, help="number of tasks")
    parser.add_argument("--initial_split", type=int, default=2, choices=[2], help="how to divide in the initial split the dataset. 2 will divide in 50%/50%")
    parser.add_argument("--valid_size", type=float, default=0.0, help="percentage of train for validation set, default not use validation set")
    parser.add_argument("--sampler", type=str, default="balanced", choices=["imbalance_sampler","balanced"], help="Select the type of sampler to ber used by dataloader. imbalance sampler is for class imbalance cases. balanced is the standard one.")
    parser.add_argument("--behaviors_check", type=str, default="yes", choices=["yes", "no"], help="Use it if we want to work with behaviors (subcategories), do not include it if not") 

    """
    Network Params
    """
    parser.add_argument("--backbone", type=str, default="movinetA0", choices=["resnet18","3dresnet18","3dresnet10","movinetA0","movinetA1","movinetA2"])
    parser.add_argument("--firsttask_modelpath", type=str, default="None", help="specify model path if start from a pre-trained model after first task")

    
    args = parser.parse_args()

    non_default_args = {
            opt.dest: getattr(args, opt.dest)
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    default_args = {
            opt.dest: opt.default
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest)
    }

    all_args = vars(args)    
    
    return args, all_args, non_default_args, default_args