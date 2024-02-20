import os
import numpy as np
import torch
import argparse
import pandas as pd
from pathlib import Path
import time
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

from Helpers.Variables import device, METRICS, WD, FILENAME_MODEL, FILENAME_HIST, FILENAME_HISTSUM, FILENAME_RES
from Helpers.Dataloader import BIODataset, BIODataLoader 
from Helpers.crl_utils import History 



def Experiment(args):
    ##### SAVE DIR 
    save_dir = os.path.join (WD,'res',  f'{args.data_type}/{args.model_type}_{args.backbone}_{args.fusion_type}/{args.postprocessor}_{args.scaling}_{args.selection_loss_type}') 
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    res_flen = str(len(os.listdir(save_dir))) 
    save_num = f'/{res_flen}/'
    RESD = save_dir + save_num
    Path(RESD).mkdir(parents=True, exist_ok=True) 

    num_fold = 4
    include_sbj = args.include_sbj

    for subj in range(1,24): 
        if subj not in include_sbj:
                continue
        
        for nf in range(1, num_fold+1):
            ts_total = pd.DataFrame(columns=METRICS)

            print('='*30)
            print(' '*4, 'START EXPERIMENT: subject{} - using fold{} as test dataset'.format(subj, nf))
            print('='*30)

            if subj < 10:
                sbj = '0' + str(subj)
            else:
                sbj = subj

            # LOAD DATA
            DATASET_DIR = '.../DMFMS/Dataset'
            Dataset_directory = f'{DATASET_DIR}/{args.data_type}'
            data_load_dir = f'{Dataset_directory}/S{sbj}/fold{nf}'
            print(f'Loaded data from --> {DATASET_DIR}/S{sbj}/fold{nf}')

            res_name = f'S{sbj}'
            nfoldname = f'fold{nf}'

            res_dir = os.path.join(RESD, res_name, nfoldname)  # 최종 저장 경로 생성
            Path(res_dir).mkdir(parents=True, exist_ok=True) 
            print(f"Saving results to ---> {res_dir}")            

            # DEFINE DATALOADER
            tr_dataset = BIODataset('train', device, data_load_dir)
            train_loader = BIODataLoader(dataset=tr_dataset, batch_size=int(args.BATCH),\
                                                num_workers=0, shuffle=True, drop_last=True)
            vl_dataset = BIODataset('valid', device, data_load_dir)
            valid_loader = BIODataLoader(dataset=vl_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            ts_dataset = BIODataset('test', device, data_load_dir)
            test_loader = BIODataLoader(dataset=ts_dataset, batch_size=int(args.BATCH), \
                                            num_workers=0, shuffle=True, drop_last=True)
            
            # DEFINE MODEL and TRAINER
            my_model = Net(args, device).to(device)

            MODEL_PATH = os.path.join(res_dir, FILENAME_MODEL)
            MODEL_PATH = f'{MODEL_PATH}'

            trainer = Trainer(args, my_model, MODEL_PATH, res_dir) 
            correctness_history = History(len(train_loader.dataset))

            # TRAIN
            tr_history = trainer.train(train_loader, valid_loader, correctness_history)
            print('End of Train\n')

            # TEST
            ts_history = trainer.eval('test', test_loader)
            print('End of Test\n')
            
            trainer.writer.close()

            # SAVE RESULTS
            trainer.save_result(tr_history, ts_history, res_dir)
            ts_total = pd.concat([ts_total, ts_history], axis=0, ignore_index=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":   
    start = time.time()
    parser = argparse.ArgumentParser(description='Dynamic multi-modal fusion for motion sickness classification')

    ### Setting hyper-parameters
    parser.add_argument('--model_type', default='DMFMS', choices=['DMFMS', 'DMFMS_woSTAM', 'DMFMS_woGate', 'DMFMS_woGate&STAM'])

    # Data hyper-parameters
    parser.add_argument('--data_type', default='MS', choices=['MS', 'Drowsy', 'Stress', 'Distraction'])
    parser.add_argument('--n_channels', default=28)
    parser.add_argument('--in_dim', default=[28,1,1,1,1], choices=[[28], [28,1], [28,1,1,1,1]], help='use for data resizing')
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--freq_time', default=750, help='frequency(250Hz)*time window(3sec.)')
    parser.add_argument('--include_sbj', default=[1,2,3,4,5,6,7,8,9,10,11,12,13], help='subject list')

    # Model's strucutre
    parser.add_argument('--backbone', default='EEGNet4', choices = ['EEGNet4', 'DeepConvNet', 'ResNet8', 'EEGConformer'])
    parser.add_argument('--fusion_type', default='average' , choices=['average', 'sum', 'concat', 'matmul']) 
    parser.add_argument('--postprocessor', default='msp', choices=['msp', 'mls', 'ebo'])
    parser.add_argument('--temp', default=10, help='temperature scaling for ebo', choices=[1.5, 0.1, 10]) 
    parser.add_argument('--scaling', default='sigmoid', choices=['softmax', 'sigmoid', 'none']) 
    parser.add_argument('--rank_weight', default=1, type=float, help='Rank loss weight') 
    parser.add_argument('--selection_loss_type', default='CE', choices=['CE', 'Focal']) 
    parser.add_argument('--CRL_user', default=True, choices=[True, False])
                                                                        
    # Training hyper-parameters
    parser.add_argument('--BATCH', default=16, help='Batch Size') 
    parser.add_argument('--EPOCH', default=100, help='Epoch') 
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer') 
    parser.add_argument('--lr', default=0.002, help='Learning Rate') 
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'])
    parser.add_argument('--step_size', default=500, help='step size for StepLR scheduler')
    parser.add_argument("--SEED", default=42)

    args = parser.parse_args()

    seed_everything(args.SEED)

    if args.model_type == 'DMFMS':
        from Models.DMFMS import Net
    elif args.model_type == 'DMFMS_woSTAM':
        from Models.DMFMS_woSTAM import Net

    if args.CRL_user == True:
        from Helpers.Trainer import Trainer
    else:
        from Helpers.Trainer_woCRL import Trainer

    # START EXPERIMENT
    Experiment(args)
