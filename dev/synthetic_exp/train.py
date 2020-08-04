#Common imports
import os
import sys
import numpy as np
import argparse
import copy
import random
import json
import pickle

#Pytorch
import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

class LinearModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
            
        super(LinearModel, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.predict_net= nn.Sequential(
		                 nn.Linear( self.inp_dim, self.out_dim),
		                 nn.Sigmoid(),
                        )
    def forward(self, x):
        return self.predict_net(x)

class NonLinearModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
            
        super(NonLinearModel, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.predict_net= nn.Sequential(
		                 nn.Linear( self.inp_dim, self.inp_dim),
		                 nn.Linear( self.inp_dim, self.out_dim),
                       )
    def forward(self, x):
        return self.predict_net(x)
    
    
def test(test_dataset, model, epoch ):
    test_acc=0.0
    test_size=0
    for test_x in enumerate(test_dataset):
        
        #Get (x, d, y) batch 
        x= test_x[1][:, :2]
        d= test_x[1][:, 2].long()
        y= test_x[1][:, 3]

        #Forward Pass
        out= model(x)
        
        test_size+= out.shape[0]
        test_acc+= torch.sum( torch.round(out).view(-1) == y).item()
        
    print(' Test Accuracy : ', 100*test_acc/test_size)
    return 100*test_acc/test_size
    
    
#Loading Train Data
# sample_case => Weighted Sampling; else uniform sampling
sample_case= int(sys.argv[1])
pretext='train_'
if sample_case:
    pretext= pretext + 'weighted'
else:
    pretext= pretext + 'uniform'

train_dataset= np.load('data/' + pretext + ' .npy')
train_dataset= torch.tensor( train_dataset ).float()
print('Train Dataset', train_dataset.shape )

#Loading Test Data
sample_case= 0
pretext='test_easy_uniform'
test_dataset_easy= np.load('data/' + pretext + ' .npy')
test_dataset_easy= torch.tensor( test_dataset_easy ).float()

pretext='test_hard_left_uniform'
test_dataset_hard_left= np.load('data/' + pretext + ' .npy')
test_dataset_hard_left= torch.tensor( test_dataset_hard_left ).float()

pretext='test_hard_right_uniform'
test_dataset_hard_right= np.load('data/' + pretext + ' .npy')
test_dataset_hard_right= torch.tensor( test_dataset_hard_right ).float()

#Torch Dataloader
n_runs=10
epochs=10
batch_size= 64
learning_rate= 0.01
feat_dim= 2
out_dim= 1
train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset_easy= torch.utils.data.DataLoader(test_dataset_easy, batch_size=batch_size, shuffle=True)
test_dataset_hard_left= torch.utils.data.DataLoader(test_dataset_hard_left, batch_size=batch_size, shuffle=True)
test_dataset_hard_right= torch.utils.data.DataLoader(test_dataset_hard_right, batch_size=batch_size, shuffle=True)

model= LinearModel(feat_dim, out_dim)
bce_loss = nn.BCELoss()
opt= optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()) }, ], 
                lr= learning_rate, weight_decay= 5e-4, momentum= 0.9,  nesterov=True )

test_acc_easy=[]
test_acc_hard_left=[]
test_acc_hard_right=[]
model_param=[]
model_bias=[]
for run in range(n_runs):
    for epoch in range(epochs):
        train_loss=0.0
        train_acc=0.0
        train_size=0
        for train_x in enumerate(train_dataset):

            opt.zero_grad()

            #Get (x, d, y) batch 
            x= train_x[1][:, :2]
            d= train_x[1][:, 2].long()
            y= train_x[1][:, 3]

            #Forward Pass
            out= model(x)
            loss= bce_loss(out, y)

            #Backward Pass
            loss.backward()
            opt.step()

            train_loss+= loss.item()
            train_size+= out.shape[0]
            train_acc+= torch.sum( torch.round(out).view(-1) == y).item()
    #         print('Batch L ', train_x[0], ' Loss : ', loss.item())

        print('Epoch :',  epoch, ' Loss : ', train_loss, ' Acc : ', train_acc/train_size)

        #Test Phase
        print('Easy Test Dataset Accuracy')
        test_acc_easy.append( test(test_dataset_easy, model, epoch) )
        print('Hard Left Test Dataset Accuracy')
        test_acc_hard_left.append(test(test_dataset_hard_left, model, epoch))
        print('Hard Right Test Dataset Accuracy')
        test_acc_hard_right.append(test(test_dataset_hard_right, model, epoch))
        print('')
    
    print('Model Parameters')
    for p in model.parameters():
        if p.requires_grad:
            if len(p.shape) ==2:
                model_param.append(p.data.detach().numpy().tolist())
            else:
                model_bias.append(p.data.detach().numpy().tolist())
            print(p.name, p.data, p.shape)

print('Final Results')
print('Test Acc Easy : ', np.mean(test_acc_easy), np.std(test_acc_easy) )
print('Test Acc Hard Left : ', np.mean(test_acc_hard_left), np.std(test_acc_hard_left) )
print('Test Acc Hard Right : ', np.mean(test_acc_hard_right), np.std(test_acc_hard_right) )

model_param= np.array(model_param)
model_bias= np.array(model_bias)
print('Model Param: ', np.mean(model_param, axis=0), np.std(model_param, axis=0) )
print('Model Bias: ', np.mean(model_bias), np.std(model_bias) )