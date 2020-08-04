#Common imports
import os
import sys
import numpy as np
import argparse
import copy
import random
import json
import pickle

# Synthetic Data Generation
# x= y*(e_c + Beta_i * e_s ) + Normal(0, Sigma_i), R^{m}, i in D
# y \pm 1 with 0.5 probability
# e_c = [1, 0]; e_s= [0, 1] ; Beta ~ U(-1, 2); Simga ~ U(0, 1) m=2; D=10  [ equally split across the range of Beta ]

def generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, pretext):
    
    #Uniform Training Data 
    data=[]
    label=[]
    domain=[]
    
    # sample_case => Weighted Sampling; else uniform sampling
    if sample_case:
        pretext= pretext + '_weighted'
    else:
        pretext= pretext + '_uniform'

    for d_idx in range(total_domains):
        
        # sample_case => Weighted Sampling; else uniform sampling
        if sample_case:
            freq= sample_freq[d_idx]
        else:
            freq= mean_freq

        #Domain Label
        d= freq*[d_idx]
        domain= domain + d

        #Class Label
        class_size= int(freq/2)
        y= class_size*[1] + (freq - class_size)*[-1]
        label= label + y

        #Data 
        for idx in range(freq):
            x= y[idx]*(e_c + beta[d_idx]*e_s) + np.random.multivariate_normal(feat_dim*[0], sigma[d_idx])
            data.append( x.tolist() )

    #     print('Freq', mean_freq, class_size)
    #     print('Domain ', d)
    #     print('Label', y)
    #     print('Data', x.tolist())

    data=np.array(data)
    label=np.array(label)
    domain=np.array(domain)

    print('Domains', np.unique(domain, return_counts=True))
    print('Labels', np.unique(label, return_counts=True))
    print('Data', data.shape)
    
    #Changing labels from -1, 1 to 0, 1 for Cross Entropy Loss
    for idx in range(label.shape[0]):
        if label[idx] == -1:
            label[idx]= 0
    print('Labels Updated :', np.unique(label, return_counts=True))

    #Concatenating (x, d, y) tuples 
    label=np.reshape(label, (label.shape[0], 1))
    domain=np.reshape(domain, (domain.shape[0], 1))
    train_dataset= np.concatenate( (data, domain, label), axis=1)    
    
    #Save dataset
    np.save('data/' + pretext + ' .npy', train_dataset)
    
e_c= np.array([1, 0])
e_s= np.array([0, 1])
feat_dim=2
total_domains=10

# Train domains
beta=np.linspace(-1, 2, total_domains)
sigma=np.zeros((total_domains, feat_dim, feat_dim))
for domain in range(total_domains):
    sigma[domain, :]= np.reshape( np.random.uniform( 0, 1, feat_dim*feat_dim ), (feat_dim, feat_dim) )

# sampling frequency per domain
lower_lim=50
upper_lim=250
sample_freq= np.random.randint(lower_lim, upper_lim , total_domains)
mean_freq= int( (lower_lim + upper_lim)/2 )
np.save('data/sample_freq.npy', sample_freq)

print('Beta: ', beta.shape, beta)
print('Sigma: ', sigma.shape, sigma[0])
print('Sample Freq: ', sample_freq.shape, sample_freq)

#Generate uniform sampling data
sample_case=0
generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, 'train')

#Generate weighted sampling data
sample_case=1
generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, 'train')

#Test Domains: Easy Set
beta=np.random.uniform(-1, 2, total_domains)
sigma=np.zeros((total_domains, feat_dim, feat_dim))
for domain in range(total_domains):
    sigma[domain, :]= np.reshape( np.random.uniform( 0, 1, feat_dim*feat_dim ), (feat_dim, feat_dim) )

print('Beta: ', beta.shape, beta)
print('Sigma: ', sigma.shape, sigma[0])
print('Sample Freq: ', sample_freq.shape, sample_freq)

#Generate uniform sampling data 
sample_case=0
generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, 'test_easy')


#Test Domains: Hard Set Right Interval
beta=np.random.uniform(2, 4, total_domains)
sigma=np.zeros((total_domains, feat_dim, feat_dim))
for domain in range(total_domains):
    sigma[domain, :]= np.reshape( np.random.uniform( 0, 1, feat_dim*feat_dim ), (feat_dim, feat_dim) )

print('Beta: ', beta.shape, beta)
print('Sigma: ', sigma.shape, sigma[0])
print('Sample Freq: ', sample_freq.shape, sample_freq)

#Generate uniform sampling data 
sample_case=0
generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, 'test_hard_right')


#Test Domains: Hard Set Left Interval
beta=np.random.uniform(-3, -1, total_domains)
sigma=np.zeros((total_domains, feat_dim, feat_dim))
for domain in range(total_domains):
    sigma[domain, :]= np.reshape( np.random.uniform( 0, 1, feat_dim*feat_dim ), (feat_dim, feat_dim) )

print('Beta: ', beta.shape, beta)
print('Sigma: ', sigma.shape, sigma[0])
print('Sample Freq: ', sample_freq.shape, sample_freq)

#Generate uniform sampling data 
sample_case=0
generate_data(feat_dim, total_domains, e_c, e_s, beta, sigma, mean_freq, sample_freq, sample_case, 'test_hard_left')
