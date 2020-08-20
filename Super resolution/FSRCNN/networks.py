from torch import nn

from torch.autograd import Variable
import torch 
import numpy as np
from constants import NUM_CLASSES, IMG_SIZE
from modules.SpatialTransformerNetwork import SpatialTransformerNetwork
from modules.FullyConnected import FullyConnected
from modules.SoftMaxClassifier import SoftMaxClassifier
from modules.Classifier import Classifier
from modules.ConvNet import ConvNet
from modules.ConvNet_after_STN import ConvNet_after_STN
from modules.vdsr import Net
from modules.utils import utils

class GeneralNetwork(nn.Module):
    def __init__(self, opt,num_channels=3, upscale_factor=1, d=64, s=12, m=4):
        super(GeneralNetwork, self).__init__()

        
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.PReLU())
        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),nn.PReLU()))
        self.mid_part = torch.nn.Sequential(*self.layers)
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

        
    def forward(self, x):
        #print('output_of_input=',x.shape)
        x=self.first_part(x)
        #print('output_of_first_layer=',x.shape)
        x=self.mid_part(x)
        #print('output_of_mid_layer=',x.shape)
        x=self.last_part(x)
        #print('output_of_mid_layer=',x.shape)
       
  
        return (x)

