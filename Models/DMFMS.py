import torch.nn as nn
import torch
import torch.nn.functional as F

from Helpers.loss_utils import FocalLoss
from .Backbone.EEGNet4 import EEGNet4



class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()

        self.args = args
        self.device = device
        self.modalities = len(args.in_dim)
        self.lossfn = torch.nn.CrossEntropyLoss() # reduction="mean" for MSLoss

        if args.backbone == 'EEGNet4': # set feature extractor for your required backbone
            self.FeatureExtractor = nn.ModuleList([EEGNet4(args, m) for m in range(self.modalities)])

        if args.data_type == 'MS':
            hidden_sizes = {"concat": 3680, "matmul": 1600, "sum": 1840, "average": 1840}
            hidden_size = hidden_sizes.get(args.fusion_type, None)
            
            self.TemporalAttn = TemporalAttn(time_step_size=46, hidden_size=46)
            self.SpatialAttn = SpatialAttn(channel_size=40, hidden_size=40)
            self.ConfidenceLayer = nn.ModuleList([nn.Linear(368, 2) for m in range(self.modalities)])
            self.MSclassifier = nn.Sequential(nn.Linear(hidden_size, self.args.n_classes))  

        if args.postprocessor == 'ebo':
            self.temperature = nn.Parameter(torch.ones(1, device=self.device) * self.args.temp)  # initialize T = original 1.5
          
    def forward(self, x, label=None, infer=False):
        data_list = data_resize(self, x)

        feat_dict = dict()
        logit_dict = dict()

        # FEATURE EXTRACTION
        for mod in range(self.modalities):
            feat_dict[mod] = self.FeatureExtractor[mod](data_list[mod]).squeeze(dim=2) 
            logit_dict[mod] = self.ConfidenceLayer[mod](feat_dict[mod].reshape(self.bs, -1)) 

        feat_list = list(feat_dict.values())
        logit_list = list(logit_dict.values())

        conf_list = []  

        # COMPUTE CONFIDENCE SCORE 
        for i in range (self.modalities):
            if self.args.postprocessor == 'msp':
                conf= torch.max(F.softmax(logit_list[i], dim=1), dim=1)[0].to(self.device) # shape: (bs,)
            
            elif self.args.postprocessor == 'mls':
                conf = torch.max(logit_list[i], dim=1)[0].to(self.device)
            
            elif self.args.postprocessor == 'ebo':
                conf = self.temperature * torch.logsumexp(logit_list[i] / self.temperature, dim=1).to(self.device)
            
            conf_list.append(conf)
        conf_stack = torch.stack(conf_list, dim=1)
        conf_b4_scale = conf_stack.clone() 
        SAMPLE_conf = torch.mean(conf_b4_scale, dim=1) # Average confidence for each modality (16,m) -> (16)
    
        # SCALING
        if self.args.scaling == 'softmax':
            confidence = F.softmax(conf_stack.squeeze(), dim=1) 
        elif self.args.scaling == 'sigmoid':
            confidence = torch.sigmoid(conf_stack.squeeze()) 

        conf_list = [confidence[:, i] for i in range(confidence.size(1))] # confidence for each modality
        conf_list = [element.unsqueeze(dim=1).unsqueeze(dim=2) for element in conf_list] 

        # GATE MECHANISM
        weight_feature_list = []
        for i in range(self.modalities):
            weight_feature =  feat_list[i].to(self.device) * conf_list[i] 
            weight_feature_list.append(weight_feature)

        # CONCATENATION OF WEIGHTED FEATURES 
        weight_feature = concat(weight_feature_list).squeeze(dim=2)  

        # BYPASSING THE STAM
        tmp_x = self.TemporalAttn(weight_feature)
        sp_x = self.SpatialAttn(weight_feature)

        if self.args.fusion_type == "concat":
            fused_feat = concat([tmp_x, sp_x])
        elif self.args.fusion_type == "sum":
            fused_feat = summation([tmp_x, sp_x])
        elif self.args.fusion_type == "average":
            fused_feat = average([tmp_x, sp_x])
        elif self.args.fusion_type == "matmul":
            fused_feat = torch.matmul(tmp_x, sp_x.transpose(1,2)) 

        fused_feat = fused_feat.reshape(self.args.BATCH, -1)
        MS_logit = F.softmax(self.MSclassifier(fused_feat), dim=1)

        if infer:
            return MS_logit
        
        # Loss function for confidence layer
        if self.args.selection_loss_type == 'CE':
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
        elif self.args.selection_loss_type == 'Focal':
            kwargs = {"alpha": 1, "gamma": 2.0, "reduction": 'none'} 
            criterion = FocalLoss(**kwargs) 
            
        label = label.squeeze().squeeze()
        MSLoss = torch.mean(self.lossfn(MS_logit, label))

        # SUMMATION LOSSES
        conf_loss_list = []
        for m in range(self.modalities): 
            conf_loss = criterion(F.softmax(logit_list[m], dim=1), label)  
            conf_loss_list.append(conf_loss)
        confidenceLoss = torch.mean(torch.stack(conf_loss_list)) 

        Loss = MSLoss + confidenceLoss # L_cls + L_conf

        return SAMPLE_conf, Loss, MS_logit 
    

    def infer(self, data_list):
        MS_logit = self.forward(data_list, infer=True)
        return MS_logit



class SpatialAttn(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(SpatialAttn, self).__init__()
        self.fc = nn.Linear(channel_size, hidden_size)

    def forward(self, weighted_features): 
        concat_feature = weighted_features
        concat_feature = concat_feature.permute(0,2,1) 
        attention_weights = F.softmax(self.fc(concat_feature))
        weighted_sp = (concat_feature * attention_weights).permute(0,2,1)
        return weighted_sp
    

class TemporalAttn(nn.Module):
    def __init__(self, time_step_size, hidden_size):
        super(TemporalAttn, self).__init__()
        self.fc = nn.Linear(time_step_size, hidden_size)

    def forward(self, weighted_features):
        concat_feature = weighted_features
        attention_weights = F.relu(self.fc(concat_feature))  
        weighted_tmp = concat_feature * attention_weights
        return weighted_tmp
        

def data_resize(self, data_list):
    new_data_list = []
    for i, dim in enumerate(self.args.in_dim):
        if i != 0:
            data_list[i] = data_list[i].unsqueeze(dim=1)
        new_data_list.append(data_list[i])  
    data_list = new_data_list

    return data_list      


def average(out_list):
    return torch.mean(torch.stack(out_list, dim=1), dim=1)

def concat(out_list):
    return torch.cat(out_list, dim=1) 

def summation(out_list):
    return torch.sum(torch.stack(out_list, dim=1), dim=1)

