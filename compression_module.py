import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BEC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p=0.2):
        x_tmp = torch.round(x * 256)
        #x_tmp = x_tmp.int()
        x_tmp = x_tmp.byte()

        p_complement = 1-p

        std = x

        binomial_noise = np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 1 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 2 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 4 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 8 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 16 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 32 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 64 + \
        np.random.binomial(1,p_complement,(std.size())).astype(np.uint8) * 128

        binomial_noise = torch.ByteTensor(binomial_noise).to(device)

        x_tmp_filter = x_tmp & binomial_noise
        x_tmp_filter = x_tmp_filter.float()
        
        x_tmp_filter = x_tmp_filter + (255.0 - binomial_noise.float()) / 2.0
        x_tmp_filter /= 255.0

        return x_tmp_filter

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class compression_module(nn.Module):
    def __init__(self,  input_channel=256, hidden_channel=128, noise=10, channel = 1,spatial = 0):
        super(compression_module, self).__init__()

        

        self.conv1 = nn.Conv2d(input_channel+1,hidden_channel,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(hidden_channel,input_channel,kernel_size = 3,stride=1,padding=1)

        self.batchnorm1 = nn.BatchNorm2d(hidden_channel)
        self.batchnorm2 = nn.BatchNorm2d(input_channel)
        
        self.conv3 = nn.Conv2d(input_channel+1,hidden_channel,kernel_size=2,stride=2)
        self.conv4 = nn.ConvTranspose2d(hidden_channel,input_channel,kernel_size=2,stride=2)
        
        self.noise = noise
        self.channel =channel
        self.spatial = spatial
               

    
    def forward(self, x):
    
               
        H = x.size()[2]
        
        C = x.size()[1]
        
        B = x.size()[0]
        
        noise_factor = torch.rand(1) * self.noise

        #noise_factor = torch.FloatTenspr([1]) * self.noise
        
        p = noise_factor.numpy()

        noise_factor = noise_factor.to(device)
        
        noise_matrix = torch.FloatTensor(np.ones((B,1,H,H))).to(device) * noise_factor
        
        x = torch.cat((x,noise_matrix),dim = 1)
        
        
        if self.spatial == 0:
            x = torch.sigmoid(self.batchnorm1(self.conv1(x)))
        
            
        elif self.spatial == 1:
            x = torch.sigmoid(self.batchnorm1(self.conv3(x)))
            
        x_tmp = x
        
        if self.channel == 'a':
            x = awgn_noise(x,noise_factor)
        elif self.channel == 'e':
            bec =  BEC.apply
            x = bec(x,p)
        elif self.channel == 'w':
            x = x
        else:
            print('error') 
               
        if self.spatial == 1:
            x = F.relu(self.batchnorm2(self.conv2(x)))
        else:
        	x = F.relu(self.batchnorm2(self.conv4(x)))

        return x
        

def awgn_noise(x, noise_factor):
    return x + torch.randn_like(x) * noise_factor

    
