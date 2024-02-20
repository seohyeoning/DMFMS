import os
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from Helpers import Utils
from Helpers.Variables import device, METRICS, FILENAME_HIST, FILENAME_HISTSUM, FILENAME_RES


class Trainer():
    def __init__(self, args, model, MODEL_PATH, res_dir):
        self.args = args
        self.model = model
        self.MODEL_PATH = MODEL_PATH
        self.set_optimizer()
        self.lossfn = nn.CrossEntropyLoss(reduction='mean')
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)
        self.writer = SummaryWriter(log_dir=res_dir) 

    def set_optimizer(self):
        if self.args.optimizer=='Adam':
            adam = torch.optim.Adam(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adam
        elif self.args.optimizer=='AdamW':
            adamw = torch.optim.AdamW(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adamw

        if self.args.scheduler == 'StepLR': 
            self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.2)
        elif self.args.scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        elif self.args.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=0, verbose=1)

    def train(self, train_loader, valid_loader, correctness_history):
        best_score = np.inf  
        history = pd.DataFrame()

        self.correctness_history = correctness_history

        for epoch_idx in range(0, int(self.args.EPOCH)):
            result = self.train_epoch(train_loader, epoch_idx) 
            history = pd.concat([history, pd.DataFrame(result).T], axis=0, ignore_index=True)

            val_result = self.eval("valid", valid_loader, epoch_idx)
            valid_loss = val_result  

            # SAVE THE BEST MODEL BASED ON VALIDATION LOSS
            if valid_loss < best_score: 
                if self.args.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else: 
                    self.scheduler.step()
                
                print(f'Validation loss decreased ({best_score:.5f} --> {valid_loss:.5f}). Saving model ...')
                best_score = valid_loss  
                torch.save(self.model.state_dict(), self.MODEL_PATH)  

            else:           
                if self.args.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else: 
                    self.scheduler.step()

        # LOAD BEST PERFORMANCE MODEL 
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        history.columns = METRICS

        return history     
       

    """ 
    EVALUATE 
    """
    def eval(self, phase, loader, epoch=0):
        self.model.eval() 
        test_history = pd.DataFrame()
        test_loss = []
        preds=[]
        targets=[]

        with torch.no_grad(): 
            for datas in loader:
                data, target = datas['data'], datas['label']
                logit = self.model.infer(data) 
                target = target.squeeze()
                test_loss.append(self.lossfn(logit, target).mean().item())
                pred = logit.argmax(dim=1,keepdim=False) 
                target = target.argmax(dim=1,keepdim=False)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        loss = sum(test_loss)/len(loader)
        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets,preds)
        f1=f1_score(targets,preds, average='macro', zero_division=0)
        preci=precision_score(targets,preds, average='macro', zero_division=0)
        recall=recall_score(targets,preds, average='macro', zero_division=0)

        print('<{}> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(phase, epoch, loss, acc*100, bacc*100, f1, preci, recall))
        
        self.write_tensorboard(phase, epoch, loss, acc, bacc, f1, preci, recall)

        if phase == 'valid':    
            return loss 
        
        elif phase=="test":
            result = [loss, acc, bacc, f1, preci, recall]
            test_history = pd.concat([test_history, pd.DataFrame(result).T], axis=0, ignore_index=True)
            test_history.columns = METRICS
            return test_history

    """ 
    TRAIN
    """
    def train_epoch(self, train_loader, epoch=0):
        self.model.train()      
        preds=[]
        targets=[]
        
        for i, datas in enumerate(train_loader):
            data, target = datas['data'], datas['label']
            idx = torch.arange(i * train_loader.batch_size, (i + 1) * train_loader.batch_size)
            confidence, cls_loss, MSlogit= self.model(data, target)
            cls_loss = cls_loss.mean() 

            ### CORRECTNESS RANKING LOSS ###
            # Make input pair for ranking
            confidence = confidence.squeeze() 
            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1) 
            idx2 = torch.roll(idx, -1)

            # Calculate target, margin for ranking
            rank_target, rank_margin = self.correctness_history.get_target_margin(idx, idx2, device)
            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1
            rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

            ranking_loss = self.ranking_criterion(rank_input1,
                                            rank_input2,
                                            rank_target)

            ranking_loss = self.args.rank_weight * ranking_loss 

            # TOTAL LOSS = L_cls + L_conf + L_cr
            loss = cls_loss + ranking_loss 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = MSlogit.argmax(dim=1)
            target = target.squeeze().argmax(dim=1)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

            _, correct = Utils.accuracy(MSlogit, target)
            self.correctness_history.correctness_update(idx, correct, MSlogit)

        # max correctness update
        self.correctness_history.max_correctness_update(epoch)

        loss = loss.item()
        acc = accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets, preds)
        f1=f1_score(targets, preds, average='macro')
        preci=precision_score(targets, preds, average='macro', zero_division=0)
        recall=recall_score(targets, preds, average='macro', zero_division=0)

        print('<train> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(epoch, loss, acc*100, bacc*100, f1, preci, recall))
        
        self.write_tensorboard('train', epoch, loss, acc, bacc, f1, preci, recall)
        
        return [loss, acc, bacc, f1, preci, recall]


        
    def save_result(self, tr_history, ts_history, res_dir):
        # save test history to csv
        res_path = os.path.join(res_dir, FILENAME_RES)
        ts_history.to_csv(res_path)
        print('Evaluation result saved')
        
        # save train history to csv
        hist_path = os.path.join(res_dir, FILENAME_HIST)
        histsum_path = os.path.join(res_dir, FILENAME_HISTSUM)
        tr_history.to_csv(hist_path)
        tr_history.describe().to_csv(histsum_path)
        print('History & History summary result saved')
        print('Tensorboard ==> \"tensorboard --logdir=runs\" \n')

    def write_tensorboard(self, phase, epoch, loss=0, acc=0, bacc=0, f1=0, preci=0, recall=0):
            if phase=='train':
                self.writer.add_scalar(f'{phase}/loss', loss, epoch)
                self.writer.add_scalar(f'{phase}/acc', acc, epoch)
            else:
                self.writer.add_scalar(f'{phase}/loss', loss, epoch)
                self.writer.add_scalar(f'{phase}/acc', acc, epoch)
                self.writer.add_scalar(f'{phase}/balanced_acc', bacc, epoch)  
                self.writer.add_scalar(f'{phase}/f1score', f1, epoch)
                self.writer.add_scalar(f'{phase}/precision', preci, epoch)
                self.writer.add_scalar(f'{phase}/recall', recall, epoch)     



    
    
