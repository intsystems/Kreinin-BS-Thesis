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

from optimizers import AdamW, AdamL2


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

class Trainer:        
    def set_wandb(self, config):
        #wandb init изменить
        self.config = config
        wandb.login(key='haha-hihi')
        self.model_name = f"{self.config['optimizer']}_lr={config['learning_rate']}_wd={config['weight_decay']}"
        
        self.run = wandb.init(project='article_tune', config=self.config, name=self.model_name)
        self.start_epoch = 0
        self.num_epochs  = self.config['epochs']
    
    def __init__(self, config):
        self.set_wandb(config)
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
        if self.config['optimizer'] == 'AdamW':
            self.optimizer = AdamW(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
            )
        elif self.config['optimizer'] == 'AdamL2':
            self.optimizer = AdamL2(
                params         = self.net.parameters(),
                lr             = self.config['learning_rate'],
                betas          = self.config['optimizer_kwargs']['betas'],
                eps            = self.config['optimizer_kwargs']['eps'],
                weight_decay   = self.config['weight_decay'],
            )
        else:
            print('There is no optimizer')
        
    def load(self, filename):
        checkpoint = torch.load(f'./checkpoint/{filename}.pth')
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net         = self.net.to(self.device)
        self.best_acc    = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        wandb.watch(self.net, log_freq=100)
    
    def wandb_log(self, config, log_config, name):    
        log_config = {
            'Epoch'               : config['i_epoch'],
            f'{name} Accuracy'    : np.mean(config['Accuracy']),
            f'{name} entropy_loss': np.mean(config['entropy_loss'])
        }
        if name == 'Train':
            for name_layer, param in self.net.named_parameters():
                opt_params = self.optimizer.get_param(param)
                for key in opt_params.keys():
                    log_config[f'{name}/{name_layer}/{key}_norm'] = torch.norm(opt_params[key], 2)
                    log_config[f'{name}/{name_layer}/{key}_min']  = torch.min(torch.abs(opt_params[key]))
                    log_config[f'{name}/{name_layer}/{key}_max']  = torch.max(torch.abs(opt_params[key]))
                    log_config[f'{name}/{name_layer}/{key}_q05']  = torch.quantile(torch.abs(opt_params[key]), 0.05)
                    log_config[f'{name}/{name_layer}/{key}_q95']  = torch.quantile(torch.abs(opt_params[key]), 0.95)
                
        wandb.log(log_config)
    
    def validate(self, i_epoch):
        # Test
        self.net.eval()
        gc.collect()
        torch.cuda.empty_cache()
        config = self.get_config()
        log_config = {}
        
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
                
            config['i_epoch'] = i_epoch
            self.wandb_log(config, log_config, name='Test')

        # Save checkpoint.
        acc = 100.*np.mean(config['Accuracy'])
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(i_epoch, self.model_name + f'_best')
    
    def train(self, i_epoch):
        # Train
        config = self.get_config()
        log_config = {}
        gc.collect()
        torch.cuda.empty_cache()
        self.net.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets, config)
            # Make backward step
            loss.backward()
            self.optimizer.step()
            log_config = self.add_log_config(log_config)
            self.optimizer.zero_grad()
            # Calculate and summary metrics
            self.metrics(outputs, targets, config)
            # LOOPA and PUPA
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(Accuracy=np.mean(config['Accuracy']), loss=np.mean(config['loss']))

        log_config['len'] = batch_idx+1
        log_config = self.mean_log_config(log_config)
        config['i_epoch'] = i_epoch
        self.wandb_log(config, log_config, name='Train')

    def add_log_config(self, log_config, name='Train'):
        for name_layer, param in self.net.named_parameters():
            opt_params = self.optimizer.get_param(param)
            for key in opt_params.keys():
                if not(f'{name}/{name_layer}/{key}' in log_config.keys()):
                    log_config[f'{name}/{name_layer}/{key}'] = torch.zeros_like(opt_params[key].detach())
                
                log_config[f'{name}/{name_layer}/{key}'] += opt_params[key].detach()
                                                             
        return log_config
    
    def mean_log_config(self, log_config, name='Train'):
        for name_layer, param in self.net.named_parameters():
            opt_params = self.optimizer.get_param(param)
            for key in opt_params.keys():
                log_config[f'{name}/{name_layer}/{key}'] = log_config[f'{name}/{name_layer}/{key}']/log_config['len']
                
        return log_config
        
    def fit(self):        
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.validate(i_epoch)
            #self.scheduler.step()
        self.run.finish()

    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.config['device'])
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        #wandb.watch(self.net, log_freq=100)
        
    def save_model(self, i_epoch, name):
        state = {
            'net'      : self.net.state_dict(), 
            'acc'      : self.best_acc, 
            'epoch'    : i_epoch, 
            'config'   : self.config
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{name}.pth')
        #print(f'Saved... Epoch[{i_epoch}]')
    
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