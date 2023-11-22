 
import torch
from .resnet18 import resnet18
from .resnet18_imagenet import resnet18_imagenet
from .resnet3d.resnet import generate_model_3dresnet
from .movinets.models import MoViNetIncDec
from .movinets.config import _C
from torch import nn
import sys 
import math
import torch.nn.functional as F 

         
class BaseModel(nn.Module):
    def __init__(self, backbone,dataset):
        super(BaseModel, self).__init__()
        self.backbone_type = backbone
        self.dataset = dataset
 
        if self.backbone_type == "resnet18" :
            if dataset == "cifar100":
                self.backbone = resnet18(avg_pool_size=4, pretrained=False)  
            elif dataset == "tiny-imagenet":
                self.backbone = resnet18(avg_pool_size=8, pretrained=False)
            elif dataset == "imagenet-subset":
                self.backbone = resnet18_imagenet()
        elif self.backbone_type == '3dresnet18':
            self.backbone = generate_model_3dresnet(18)
        elif self.backbone_type == '3dresnet10':
            self.backbone = generate_model_3dresnet(10)
        elif self.backbone_type == 'movinet':
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA0, causal = False, pretrained = False)
        else:
            sys.exit("Model Not Implemented")

        self.heads = nn.ModuleList()
    
    def get_feat_size(self):
        return self.backbone.feature_space_size

    def add_classification_head(self, n_out):
        #TODO: qui forse fare che se il modello è per DataIncDec, forse una sola testa, non più...
        if self.backbone_type == "movinet":
            self.heads.append(
                torch.nn.Sequential(self.backbone.add_head(num_classes=n_out)))
        else:
            self.heads.append(
                torch.nn.Sequential(nn.Linear(self.backbone.feature_space_size, n_out, bias=False)))

    
    def reset_backbone(self, backbone = None):

        if self.dataset == "cifar100":
            self.backbone = resnet18(avg_pool_size=4, pretrained=False)  
        elif self.dataset == "tiny-imagenet":
            self.backbone = resnet18(avg_pool_size=8, pretrained=False)  
        
        elif self.dataset == "imagenet-subset":
            self.backbone = resnet18_imagenet()
        elif self.dataset == "kinetics" or backbone == "3dresnet18":
             self.backbone == generate_model_3dresnet(18)
        elif self.dataset == "kinetics" or backbone == "movinet":
             self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA0, causal = False, pretrained = False)
        

    def forward(self, x):
        results = {}
        features = self.backbone(x)

        if self.backbone_type == 'movinet':
            for id, head in enumerate(self.heads):
                x = head(features)
                results[id] = x.flatten(1)
        else:
            for id, head in enumerate(self.heads):
                results[id] = head(features)
        
        return results, features
    
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False
    

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
 
    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters(): 
                    param.requires_grad=False
               
