import torch
from tqdm import tqdm
from copy import deepcopy

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator_incdec import MetricEvaluatorIncDec
 

#TODO: vedere se ereditare da IncrementalApproach ha senso e se modificare qualcosa
class DataIncrementalDecrementalMethod(IncrementalApproach):
    #TODO: modificare init, non necessito di class_per_task probabilmente, non so di task_dict
    
    def __init__(self, args, device, out_path, task_dict, total_classes,behaviors_per_task, behavior_dicts):
        #TODO: class_per_task, come passarlo
        self.total_classes = total_classes
        super().__init__(args, device, out_path, total_classes, task_dict)
        #TODO: vedere se da BaseModel necessito di modificare qualcosa in caso

        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        #TODO: forse modificare come aggiungere head, tanto la si crea una sola volta
        self.model.add_classification_head(self.total_classes)
        self.print_running_approach()


    def print_running_approach(self):
        super(DataIncrementalDecrementalMethod, self).print_running_approach()
        
    #TODO: modifica info necessarie per impostare parametri per il pretraining
    def pre_train(self,  task_id, trn_loader, test_loader):
        #TODO: aggiunta della classification head non penso debba dipendere dal task_id, siccome
        #il numero di classi non cambia
        #self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        # necessary only for tsne 
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        self.old_model.to(self.device)
        
        #TODO: modificare, forse non prendere quello precedente e cambiare direttamente la logica
        #perchè 2 tipi di approcci diversi
        #TODO: forse da non modificare nulla, ma non so per la logica delle teste multiple...
        super(DataIncrementalDecrementalMethod, self).pre_train(task_id)

    #TODO: forse non necessario da cambiare, controllare...
    def train(self, task_id, train_loader, epoch, epochs):
        self.model.to(self.device)
        self.model.train()
        
       
        train_loss, n_samples = 0, 0
        for images, targets, _ in  tqdm(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            current_batch_size = images.shape[0]
            n_samples += current_batch_size
 

            outputs, _ = self.model(images)
            
            #TODO: verificare che criterion debba dipendere da task_id, forse si per poterlo salvare/fare accorgimenti diversi
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss * current_batch_size
        
        self.train_log(task_id, epoch, train_loss/n_samples)  

    #TODO: t è il task_id, immagino di doverlo usare poi per salvare data del task
    #cross entropy correct for our task of video classification
    def criterion(self, outputs, targets):
        #TODO: rescale targets non dovrebbe servire...
        #targets = self.rescale_targets(targets, t)
        #here is 0 cause we only have a head
        return torch.nn.functional.cross_entropy(outputs[0], targets)
        
        
        
    def post_train(self, task_id, trn_loader=None):
        pass 

    #TODO: definire evaluation step
    #TODO: definire quindi accuracy e AP
    #TODO: accuracy e AP vanno probabilmente generate per ogni classe e per ogni sub-behavior, 
    # per identificare comportamenti anomali al variare dei dati e dei training
    #TODO: controllare https://torchmetrics.readthedocs.io/en/stable/classification/average_precision.html
    #magari ci sono già implementati metodi interessanti
    def eval(self, current_training_task, test_id, loader, epoch, verbose):
        #TODO: modificare anche metric evaluator per gestire anche AP e differenziazione tra classi...
        metric_evaluator = MetricEvaluatorIncDec(self.out_path, self.task_dict, self.total_classes)
        
        cls_loss, n_samples = 0, 0 
        with torch.no_grad():
            self.model.eval()
            for images, targets, _  in tqdm(loader):
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
 
                outputs, features = self.model(images)
                _, old_features = self.old_model(images)
                
                cls_loss += self.criterion(outputs, targets) * current_batch_size
                 

                metric_evaluator.update(targets, 
                                        self.compute_probabilities(outputs, 0))

            #task aware accuracy e task agnostic accuracy
            acc, ap, acc_per_class = metric_evaluator.get(verbose=verbose)
 
              
            self.log(current_training_task, test_id, epoch, cls_loss/n_samples, acc)          
            
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))

            return acc, ap, cls_loss/n_samples, acc_per_class
        
    #TODO: definire log da fare...
    def log(self, current_training_task, test_id, epoch, cls_loss , acc):
        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_accuracy"
        self.logger.add_scalar(name_tb, acc, epoch)

    def train_log(self, current_training_task, epoch, cls_loss):
        name_tb = "training_task_" + str(current_training_task) + "/training_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)


    def compute_probabilities(self, outputs, head_id):
      probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
      return probabilities