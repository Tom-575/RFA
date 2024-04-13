import argparse
import logging
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import linalg as LA
import random
import os
from typing import Dict, List
import json
import copy
import sys


from networks.wideresnet import WideResNet

from utils.logger import Logger
from utils.attack_tools import *

class VAE_MLP(nn.Module):
    def __init__(self, layer_args, latent_dim: int = 512, hidden_dims: List = None, is_ae = False, kl_coef=1, tc_coef=6.0, bias=True, **kwargs) -> None:
        super(VAE_MLP, self).__init__()
        self.layer_args = layer_args
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()
        self.vae_layer_args = []
        self.is_ae = is_ae

        enc_modules = []
        if hidden_dims is None:
            hidden_dims = [1024, 512]

        in_c = self.layer_args[0] * self.layer_args[1] * self.layer_args[2]
        # Build Encoder
        for h_dim in hidden_dims:
            enc_modules.append(
                nn.Sequential(
                    nn.Linear(in_c, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=False)
                )
            )
            in_c = h_dim

        # Mid
        self.mlp_mu = nn.Linear(in_c, latent_dim)
        self.mlp_var = nn.Linear(in_c, latent_dim)
        in_c = latent_dim

        # Build Decoder
        dec_modules = []
        hidden_dims.reverse()
        for h_dim in hidden_dims:
            dec_modules.append(
                nn.Sequential(
                    nn.Linear(in_c, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=False)
                )
            )
            in_c = h_dim
        dec_modules.append(nn.Linear(in_c, self.layer_args[0] * self.layer_args[1] * self.layer_args[2]))

        self.enc = nn.Sequential(*enc_modules)
        self.dec = nn.Sequential(*dec_modules)


    def encode(self, x):
        x = x.view(x.size(0), -1)
        f = self.enc(x)
        mu, logvar = self.mlp_mu(f), self.mlp_var(f)
        return mu, logvar

    def decode(self, z):
        out = self.dec(z)
        out = out.reshape(-1, self.layer_args[0] , self.layer_args[1] , self.layer_args[2])
        return out

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.is_ae:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        out = self.decode(z)
        
        return out, mu, logvar, z
    
    @staticmethod
    def get_loss(x_in, x_out, mu, logvar, alpha=1.0, alpha_kl=3.0):
        kl_loss = ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / ( logvar.size(0) * logvar.size(1) )
        mse_loss = F.mse_loss(x_in, x_out)

        return alpha *  mse_loss + alpha_kl * kl_loss


class WRN_with_RFA(nn.Module):
    def __init__(self, base_model, vae_r, num_classes=10):
        super(WRN_with_RFA, self).__init__()
        self.base_model = base_model
        self.ae_r = vae_r

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.block1(x)
        x = self.base_model.block2(x)
        x = self.base_model.block3(x)
        tmp_size =  x.size()
        x, _, _, _ = self.ae_r(x.view(x.size(0),-1))
        x = x.view(*tmp_size)

        x = self.base_model.relu(self.base_model.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.base_model.nChannels)
        out = self.base_model.fc(x)
        
        return out

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def logits_accuracy_confidence(output, target):
    _, pred = output.topk(1, 1, True, True)
    correct = pred.eq(target.view(-1, 1))
    accuracy = correct.sum().float() / target.size(0)
    
    probs = F.softmax(output, dim=1)
    confidences = []
    for i in range(target.size(0)):
        confidence = probs[i, target[i]].item()
        confidences.append(confidence)
    mean_confidence = sum(confidences) / len(confidences)
    
    return accuracy*100.0, mean_confidence*100.0

def get_aes(x, y, attack, attack_name):
    if attack_name[:attack_name.index('(')] in ["Clean", "AALinfAttack", "AAL2Attack", "JPLinfAttack", "AALinfAttack", "ReColorAttack", "STAAttack"]:
        xa = attack(x, y)
    else:
        xa = attack.perturb(x)
        # xa = attack.perturb(x, y)
    return xa

def eval_robustness(args, model, testloader, attack_names, attacks):
    model.eval()
    acc = {attack_name: AverageMeter() for attack_name in attack_names}
    
    for batch_index, (inputs, labels) in enumerate(testloader):
        if args.test_num != None and batch_index >= args.test_num:
            break

        print(f'BATCH {batch_index:05d}')
        if torch.cuda.is_available():
            x, label = inputs.cuda(), labels.cuda()
            
        for attack_name, attack in zip(attack_names, attacks):
            model.eval()

            model.zero_grad()
            if x.grad != None:
                x.zero_grad()
            xa = get_aes(x, label, attack, attack_name)
            torch.cuda.empty_cache()
            model.zero_grad()

            with torch.no_grad():
                out = model(xa)
            
            y = label
            prec1, _ = logits_accuracy_confidence(out, y)
            acc[attack_name].update( prec1*len(y), len(y) )
            
            print("| Test Batch #%d\t %s \tAcc: %.3f%%;" % (batch_index, attack_name, acc[attack_name].avg))


def parser_eval():
    parser = argparse.ArgumentParser(description='WRN_RFA')
    parser.add_argument('--RFA_model_ckpt', type=str, default='')
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack function names')
    parser.add_argument('--test_num', default=10000000, type=int, help='the number of test batch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_classes', default=10, type=int)

    return parser

if __name__ == "__main__":
    # # Setup
    print('[Phase 1] : Setup')
    parse = parser_eval()
    args = parse.parse_args()

    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'CIFAR10':
        testset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform_test)
        args.num_classes = 10
    elif args.dataset == 'CIFAR100':
        testset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False, download=True, transform=transform_test)
        args.num_classes = 100
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print('[Phase 2] : Build Model')
    model = WideResNet(28, args.num_classes, widen_factor=10, dropRate=0, activation='ReLU')
    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]
    d_layer_arg = [640, 8, 8]    
    hidden_dims = [512, 256] #
    latent_dim = 128 
    vae_r = VAE_MLP(d_layer_arg, latent_dim=latent_dim, hidden_dims=hidden_dims.copy(), is_ae=False).cuda()
    model = WRN_with_RFA(model, vae_r, num_classes=args.num_classes).cuda()
    if args.RFA_model_ckpt != '':
        model.load_state_dict(torch.load(args.RFA_model_ckpt))    
    print("Load Model Success!")
    model.eval()
    eval_robustness(args, model, test_loader, attack_names, attacks)





