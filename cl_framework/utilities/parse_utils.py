import argparse
from ast import parse
import sys
    
def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--approach", type=str,default="lwf", choices=["finetuning", "ewc","lwf","incdec"])
    parser.add_argument("--baseline", action='store_true') 
    parser.add_argument("--imbalanced", action='store_true') 
    parser.add_argument("--outpath", "-op",default="./", type=str) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nw", type=int, default=2, help="num workers for data loader")
    parser.add_argument("--freeze_bn", type=str, default="no")

    
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
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_first_task", type=float, default=1e-3, help="for tiny-imagenet and cifar100, for imagenet-subet default  are in IncrementalApproach.py")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="fixd", choices=["fixd", "multi_step"])
    parser.add_argument("--device", type=int, default=0)
    
    "Dataset Settings"
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100","tiny-imagenet","imagenet-subset", "kinetics"], help="dataset to use") 
    #parser.add_argument("--data_path",type=str, default="/Kinetics",help="path where imagenet subset and tiny-imagenet are saved")
    parser.add_argument("--data_path",type=str, default="/cl_data",help="path where imagenet subset and tiny-imagenet are saved")
    parser.add_argument("--n_class_first_task", type=int, default=50, help="if greater than -1 use a larger number of classes for the first class, n_task include this one")
    parser.add_argument("--n_task", type=int, default=6, help="number of task")
    parser.add_argument("--initial_split", type=int, default=2, help="how to divide in the initial split the dataset. 2 will divide in 50%/50%")
    parser.add_argument("--valid_size", type=float, default=0.0, help="percentage of train for validation set, default not use validation set")
    """
    Network Params
    """
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","3dresnet18","3dresnet10","movinet"])
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