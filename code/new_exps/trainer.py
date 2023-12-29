# Torch import
import torch
import math
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Stuff
import pickle
import gc
import os, glob
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18
from optim_fancy import *
transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

class Trainer:        
    def set_wandb(self):
        #wandb init изменить
        self.config = {
                'architecture'    : 'ResNet18',
                'optimizer'       : self.name_optimizer,
                'learning_rate'   : 3e-5,
                'weight_decay'    : 1e-6,
                'params_percent'  : self.percent,
                'optimizer_kwargs': {'betas': (0.9, 0.999), 'eps': 1e-7},
                'scheduler_name'  : 'CosineAnnealingLR',
                'scheduler_kwargs': {'eta_min': 2e-4, 'T_max': 300, 'gamma': 0.85,'step_size': 7, },
                'epochs'          : 30,
                'batch_size'      : 128
            }
        wandb.login(key='84d6a92704bf4bf19d2ecc87a55eea5ce77a8725')
        self.model_name = self.config['optimizer']+ f"_[{self.config['params_percent']}]"
        self.run = wandb.init(project='new_article_exps', config=self.config, name=self.model_name)
        self.start_epoch = 0
        self.num_epochs  = self.config['epochs']
    
    def __init__(self, name_optimizer, percent):
        self.name_optimizer = name_optimizer
        self.percent = percent
        self.set_wandb()
        self.set_net()
        self.set_opt_sched()
        # Create dataset and datalodaer for train
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = DataLoader(trainset, batch_size=self.config['batch_size'], shuffle=True, num_workers=10)
        # Create dataset and dataloader for validation
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = DataLoader(testset, batch_size=self.config['batch_size'], shuffle=False, num_workers=10)
        # Initialize all essentials critetia and metrics
        self.entropy  = torch.nn.CrossEntropyLoss()
        self.best_acc = 0
    
    def criterion(self, outputs, targets, config):
        loss_entropy = self.entropy(outputs, targets)
        config['entropy_loss'].append(loss_entropy.item())
        config['loss'].append(loss_entropy.item())
        return loss_entropy
    
    def set_opt_sched(self):
        if self.name_optimizer == 'AdamW_batch_batch':
            self.optimizer = AdamW_batch_batch(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
                params_percent = self.config['params_percent']
                )
        if self.name_optimizer == 'AdamW_batch_g':
            self.optimizer = AdamW_batch_g(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
                params_percent = self.config['params_percent']
                )
        if self.name_optimizer == 'AdamW_sega_batch':
            self.optimizer = AdamW_sega_batch(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
                params_percent = self.config['params_percent']
                )
        if self.name_optimizer == 'AdamW_sega_g':
            self.optimizer = AdamW_sega_g(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
                params_percent = self.config['params_percent']
                )
        if self.name_optimizer == 'AdamW':
            self.optimizer = AdamW(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
                params_percent = self.config['params_percent']
                )
        #self.scheduler = torch.optim.lr_scheduler.StepLR(
            #self.optimizer, 
            #step_size=self.config['scheduler_kwargs']['step_size'],
            #gamma=self.config['scheduler_kwargs']['gamma']
            #)
        
    def load(self, filename):
        checkpoint = torch.load(f'./checkpoint/{filename}.pth')
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net         = self.net.to(self.device)
        self.best_acc    = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        wandb.watch(self.net, log_freq=100)
    
    def wandb_log(self, config, name):    
        wandb.log({f'{name} entropy_loss': np.mean(config['entropy_loss']),
                   f'{name} Accuracy'    : np.mean(config['Accuracy']),
                   'Epoch'               : config['i_epoch']})
    
    def validate(self, i_epoch):
        # Test
        self.net.eval()
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        
        with torch.no_grad():
            loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
            for batch_idx, (inputs, targets) in loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, config)
                # Calculate and summary metrics
                self.metrics(outputs, targets, config)
                # LOOPA and PUPA
                loop.set_description(f"Epoch (Test)[{i_epoch}/{self.num_epochs}]")
                loop.set_postfix(Accuracy=np.mean(config['Accuracy']), loss=np.mean(config['loss']))
                gc.collect()
                torch.cuda.empty_cache()
                
            config['i_epoch'] = i_epoch
            self.wandb_log(config, name='Test')

        # Save checkpoint.
        acc = 100.*np.mean(config['Accuracy'])
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(i_epoch, self.model_name + f'_best')
    
    def train(self, i_epoch):
        # Train
        config = self.get_config()
        self.net.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets, config)
            gc.collect()
            torch.cuda.empty_cache()
            # Make backward step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Calculate and summary metrics
            self.metrics(outputs, targets, config)
            # LOOPA and PUPA
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(Accuracy=np.mean(config['Accuracy']), loss=np.mean(config['loss']))
            gc.collect()
            torch.cuda.empty_cache()
        config['i_epoch'] = i_epoch
        self.wandb_log(config, name='Train')

    def fit(self):        
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.validate(i_epoch)
            #self.scheduler.step()
        self.run.finish()

    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(1)
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        #wandb.watch(self.net, log_freq=100)
        
    def save_model(self, i_epoch, name):
        state = {
            'net'      : self.net.state_dict(), 
            'acc'      : self.best_acc, 
            'epoch'    : i_epoch, 
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{name}.pth')
        print(f'Saved... Epoch[{i_epoch}]')
    
    def get_config(self):
        return {
            'loss'        : [],
            'entropy_loss': [],
            'i_epoch'     : 0,
            'Accuracy'    : [],
            }
    
    def metrics(self, outputs, targets, config):
        _, predicted = outputs.max(1)
        config['Accuracy'].append(predicted.eq(targets).sum().item()/targets.size(0))