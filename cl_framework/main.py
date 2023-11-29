from utilities.generic_utils import experiment_folder, result_folder, \
                            get_task_dict, seed_everything, rollback_model, \
                            store_model, store_valid_loader, get_class_per_task, remap_targets, get_behaviors_per_task, \
                            get_task_dict_incdec, AverageMeter
from utilities.parse_utils import get_args
from utilities.matrix_logger import Logger, IncDecLogger
from torch.utils.data.dataloader import DataLoader

# approaches 
from continual_learning.FinetuningMethod import FineTuningMethod
from continual_learning.DataIncrementalDecrementalMethod import DataIncrementalDecrementalMethod
 
from continual_learning.LearningWithoutForgetting import LWF


# dataset 
from dataset.continual_learning_dataset import ContinualLearningDataset
from dataset.data_inc_dec_dataset import DataIncDecBaselineDataset
from dataset.dataset_utils import get_dataset 
import sys 

from utilities.summary_logger import SummaryLogger
import os 
from copy import deepcopy
import math
import time
 

if __name__ == "__main__":
    
    # args
    args, all_args, non_default_args, all_default_args = get_args()
    
    print(args.outpath)
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
        
    # device
    device = "cuda:" + str(args.device)
    
     
    # if True create folder exp_folder, else create folder with the name of the approach
    dev_mode = False

    stop_first_task = args.stop_first_task

    # generate output folder and result folders
    out_path, exp_name = experiment_folder(args.outpath, dev_mode, args.approach)
 

    print("Current Seed {}".format(args.seed))
    
    # fix seed
    seed_everything(seed=args.seed)
    
    """
    Dataset Preparation
    """

    #TODO: completare e controllare funzioni tutto come desiderato
    train_set, test_set, validation_set, total_classes = get_dataset(args.dataset, args.data_path)
    
    
    # mapping between classes and shuffled classes and re-map dataset classes for different order of classes
    #TODO: non so se utile shuffle le label... per ora eliminato
    if not (args.approach == 'incdec'): 
        train_set, test_set, label_mapping = remap_targets(train_set, test_set, total_classes)
    
     
    # class_per_task: number of classes not in the first task, if the first is larger, otherwise it is equal to total_classes/n_task
    if args.approach == 'incdec':
        #TODO: to implement oltre la baseline...
        behaviors_per_task = get_behaviors_per_task(total_classes, args.n_task, args.baseline)
    else:
        class_per_task = get_class_per_task(args.n_class_first_task, total_classes, args.n_task)
    
    
    # task_dict = {task_id: list_of_class_ids}
    if args.approach == 'incdec':
        #TODO: aggiungere logica oltre la baseline
        task_dict, behavior_dicts = get_task_dict_incdec(args.n_task, total_classes, behaviors_per_task, args.baseline)
    else:
        task_dict = get_task_dict(args.n_task, total_classes, class_per_task, args.n_class_first_task)   
    
    #TODO: fare print diverso per diverso tipo di task del DataIncDec... per ora commento
    """ print("Dataset: {}, N task: {}, Large First Task Classes: {}, Classes Per Task : {}".format(args.dataset,
                                                                                                  args.n_task,
                                                                                                  args.n_class_first_task,
                                                                                                  class_per_task)) """
        
    """
    Generate Subset For Each Task
    """

    if args.approach == 'incdec':
        if args.baseline:
            cl_train_val = DataIncDecBaselineDataset(train_set, task_dict,  
                                                    args.n_task, args.initial_split, 
                                                    total_classes,
                                                    train=True, validation=validation_set,
                                                    valid_size=args.valid_size)
        else:
            #TODO: to be implemented the logic over the baseline
            print('Still not implemented...')
    else:
        cl_train_val = ContinualLearningDataset(train_set, task_dict,  
                                                args.n_task, args.n_class_first_task, 
                                                class_per_task,total_classes,
                                                valid_size=args.valid_size, train=True)

    
    train_dataset_list, train_sizes, val_dataset_list, val_sizes = cl_train_val.collect()

    if args.approach == 'incdec':
        if args.baseline:
            cl_test = DataIncDecBaselineDataset(test_set, task_dict,  
                                                    args.n_task, args.initial_split, 
                                                    total_classes,
                                                    train=False, validation=None,
                                                    valid_size=None,)
        else:
            #TODO: to be implemented the logic over the baseline
            print('Still not implemented...')
    else:
        cl_test = ContinualLearningDataset(test_set, task_dict,  
                                        args.n_task, args.n_class_first_task, 
                                        class_per_task,total_classes,
                                        train=False)

    test_dataset_list, test_sizes, _, _  = cl_test.collect()
    #test_loaders = [DataLoader(test, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw) for test in test_dataset_list]
    test_loaders = [DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.nw) for test in test_dataset_list]


    train_loaders = []
    if args.valid_size > 0 or validation_set != None:
        if args.sampler == 'imbalance_sampler':
            for train in train_dataset_list:
                sampler = cl_train_val.get_weighted_random_sampler(train.indices)
                train_loaders.append(DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.nw, sampler=sampler))
        else:
            train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw) for train in train_dataset_list]
        print("Creating Validation Set")
        
        #valid_loaders = [DataLoader(valid, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw) for valid in val_dataset_list]
        valid_loaders = [DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.nw) for valid in val_dataset_list]
    
    else:
        print("Not using Validation")
        if args.sampler == 'imbalance_sampler':
            for train in train_dataset_list:
                sampler = cl_train_val.get_weighted_random_sampler(train.indices)
                train_loaders.append(DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.nw, sampler=sampler))
        else:
            train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw) for train in train_dataset_list]
        valid_loaders = test_loaders
 

    
    """
    Logger Init
    """
    if args.approach == 'incdec':
        logger = IncDecLogger(out_path=out_path, n_task=args.n_task, task_dict=task_dict, test_sizes=test_sizes, num_classes=total_classes)
    else:
        logger = Logger(out_path=out_path, n_task=args.n_task, task_dict=task_dict, test_sizes=test_sizes)
    result_folder(out_path, "tensorboard")
    result_folder(out_path, "logger")

    #Average time keeper for training
    avg_time_train = AverageMeter()
 
 

    if args.approach == 'finetuning':
        approach = FineTuningMethod(args=args, device=device, out_path=out_path,
                                    class_per_task=class_per_task, 
                                    task_dict=task_dict )

    elif args.approach == 'lwf':
        approach = LWF(args=args, device=device, out_path=out_path, 
                       class_per_task=class_per_task,
                       task_dict=task_dict )
        
    elif args.approach == 'incdec':
        approach = DataIncrementalDecrementalMethod(args=args, device = device,
                    out_path=out_path,
                    task_dict=task_dict,
                    total_classes=total_classes,
                    behaviors_per_task=behaviors_per_task,
                    behavior_dicts = behavior_dicts,
                    )
    else:
        sys.exit("Approach not Implemented")

    summary_logger = SummaryLogger(all_args, all_default_args, args.outpath, args.approach)
    summary_logger.summary_parameters(exp_name)
 
    for task_id, train_loader in enumerate(train_loaders):


        if  task_id == 0 and args.firsttask_modelpath != "None":
 
            approach.pre_train(task_id, train_loader,  valid_loaders[task_id])
 
            print("Loading model from path {}".format(os.path.join(args.firsttask_modelpath, "{}_seed_{}".format(args.dataset, args.seed), "0_model.pth")))

            rollback_model(approach, os.path.join(args.firsttask_modelpath, "{}_seed_{}".format(args.dataset, args.seed),"0_model.pth"), device, name=str(task_id))
  
            
            epoch = 100
        else:
 
            
            print("#"*40 + " --> TRAINING HEAD/TASK {}".format(task_id))

            """
            Pre-train
            """
            

            approach.pre_train(task_id, train_loader,  valid_loaders[task_id])

            
            best_taw_accuracy,  best_tag_accuracy, best_accuracy, best_mAP = 0, 0, 0, 0
            best_epoch = 0
            best_loss = math.inf 
            
            if task_id == 0 and args.dataset == "imagenet-subset":
                n_epochs = 160 # default number of epochs for imagenet subset 
            else:
                n_epochs = args.epochs
            
            
            """
            Main train Loop
            """
            #for early stopping when the validation loss doesn't improve
            no_decrement_count = 0
            best_loss = float(math.inf)
                    
            for epoch in range(n_epochs):
                print("Epoch {}/{}".format(epoch, n_epochs))


                
                if epoch == 0:
                    store_model(approach, out_path)

                end_time = time.time()

                approach.train(task_id, train_loader, epoch, n_epochs)
                
                if args.approach == 'incdec':
                    acc, _ , test_loss, _, mean_ap_eval,_ = approach.eval(task_id, task_id, valid_loaders[task_id], epoch, verbose=True, testing=False)
                else:
                    taw_acc, tag_acc, test_loss  = approach.eval(task_id, task_id, valid_loaders[task_id], epoch,  verbose=True)
                
                previous_lr = approach.optimizer.param_groups[0]["lr"]
                
                if args.scheduler_type == 'reduce_plateau':
                    approach.reduce_lr_on_plateau.step(mean_ap_eval)
                else:
                    approach.reduce_lr_on_plateau.step()
                    
                current_lr = approach.optimizer.param_groups[0]["lr"]
                
                if args.approach == 'incdec':
                    if mean_ap_eval > best_mAP:
                        old_mAP = best_mAP
                        best_mAP = mean_ap_eval
                        name_model = "best_mAP_task_" + str(task_id)
                        store_model(approach, out_path,name=name_model)
                        print(f"  --> from mAP {old_mAP:.3f} to {best_mAP:.3f}")
                        best_epoch = epoch
                else:
                    if taw_acc > best_taw_accuracy:
                        old_accuracy = best_taw_accuracy
                        best_taw_accuracy = taw_acc
                        store_model(approach, out_path)
                        print(f"  --> from acc {old_accuracy:.3f} to {taw_acc:.3f}")

                
                

                avg_time_train.update(time.time() - end_time)

                print(f"Last time {avg_time_train.val:.3f} - Average time ({avg_time_train.avg:.3f})\t")

                if current_lr != previous_lr:
                    rollback_model(approach, os.path.join(out_path,"best_mAP_task_{}_model.pth").format(task_id), device, name=str(task_id))

                #Commented because for now i'll do a early stopping when the lr becomes lower than a threshold
                """ 
                #checks if the mAP has decreased or not
                if mean_ap_eval < best_mAP:
                    no_decrement_count = 0
                    best_mAP = mean_ap_eval
                else:
                    no_decrement_count += 1
                #early stops if too many epochs without improving
                if no_decrement_count == args.early_stopping_val:
                    print(f"Early stopping because classification loss didn't improve for{args.early_stopping_val} epochs\t")
                    break """
                #Stops if the learning rate is lower than a threshold
                print(f"Current learning rate for the next epoch is: {current_lr}")
                if current_lr <= float(1e-6):
                    print(f"Early stopping because learning rate threshold is reached \t")
                    break

            logger.print_best_epoch(best_epoch, task_id)
            
            
        """
        Test Final Model
        """
    
        store_model(approach, out_path, name=str(task_id))
 
        

        for test_id in range(task_id + 1):
            
            if args.approach == 'incdec':
                acc_value, ap_value, _, acc_per_class, mean_ap, map_weighted  = approach.eval(task_id, test_id, test_loaders[test_id], epoch,  verbose=False, testing=True)
                logger.update_accuracy(current_training_task_id=task_id, test_id=test_id, acc_value=acc_value, ap_value=ap_value, acc_per_class=acc_per_class, mean_ap=mean_ap, map_weighted=map_weighted)
                #TODO: questo è forse per misurare quando si dimentica dei vecchi task, introdurre qualche metrica del genere
                if test_id < task_id:
                    logger.update_forgetting(current_training_task_id=task_id, test_id=test_id)
                logger.print_latest(current_training_task_id=task_id, test_id=test_id)
            else:
                acc_taw_value, acc_tag_value, _,  = approach.eval(task_id, test_id, test_loaders[test_id], epoch,  verbose=False)                                                                                                            
                logger.update_accuracy(current_training_task_id=task_id, test_id=test_id, acc_taw_value=acc_taw_value, acc_tag_value=acc_tag_value)
                if test_id < task_id:
                    logger.update_forgetting(current_training_task_id=task_id, test_id=test_id)
                logger.print_latest(current_training_task_id=task_id, test_id=test_id)

 
        """
        Post Training
        """
        logger.compute_average()
        logger.print_file()
  
        approach.post_train(task_id=task_id, trn_loader=train_loader)

        #If i want to stop at the first task
        if stop_first_task:
            break

    
    summary_logger = SummaryLogger(all_args, all_default_args, args.outpath, args.approach)
    summary_logger.update_summary(exp_name, logger, avg_time_train.avg)
    store_valid_loader(out_path, valid_loaders, False)