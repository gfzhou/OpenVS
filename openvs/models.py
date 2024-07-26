import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from openvs.args import VanillaModelArgs

class VanillaNet(nn.Module):
    def __init__(self, args: VanillaModelArgs ):
        super().__init__()
        nBits = args.nBits
        nnodes = args.nnodes
        dropoutfreq = args.dropout
        if args.dataset_type == "binaryclass":
            self.classification = True
        else:
            self.classification = False
        self.fc1 = nn.Linear(nBits, nnodes)
        self.fc2 = nn.Linear(nnodes, nnodes)
        self.fc3 = nn.Linear(nnodes, nnodes)
        self.fc_out = nn.Linear(nnodes, 1)
        self.bn1 = nn.BatchNorm1d(num_features=nnodes)
        self.bn2 = nn.BatchNorm1d(num_features=nnodes)
        self.bn3 = nn.BatchNorm1d(num_features=nnodes)
        self.dropout1 = nn.Dropout(dropoutfreq)
        self.dropout2 = nn.Dropout(dropoutfreq)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        if self.classification:
            x = self.out_activation(x)
        return x

class VanillaNet2(nn.Module):
    def __init__(self, args: VanillaModelArgs ):
        super().__init__()
        nBits = args.nBits
        nnodes = args.nnodes
        dropoutfreq = args.dropout
        if args.dataset_type == "binaryclass":
            self.classification = True
        else:
            self.classification = False
        self.fc1 = nn.Linear(nBits, nnodes)
        self.fc2 = nn.Linear(nnodes, nnodes)
        self.fc3 = nn.Linear(nnodes, nnodes)
        self.fc_out = nn.Linear(nnodes, 1)
        self.bn1 = nn.BatchNorm1d(num_features=nnodes)
        self.bn2 = nn.BatchNorm1d(num_features=nnodes)
        self.bn3 = nn.BatchNorm1d(num_features=nnodes)
        self.dropout = nn.Dropout(dropoutfreq)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        if self.classification:
            x = self.out_activation(x)
        return x

class VanillaNet3(nn.Module):
    def __init__(self, args: VanillaModelArgs ):
        super().__init__()
        nBits = args.nBits
        nnodes = args.nnodes
        dropoutfreq = args.dropout
        self.nlayers = args.nlayers
        if args.dataset_type == "binaryclass":
            self.classification = True
        else:
            self.classification = False
        self.fc_in = nn.Linear(nBits, nnodes)
        self.fcs = nn.ModuleList([nn.Linear(nnodes, nnodes) for i in range(self.nlayers)] )
        self.fc_out = nn.Linear(nnodes, 1)
        
        self.bn1 = nn.BatchNorm1d(num_features=nnodes)
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=nnodes) for i in range(self.nlayers)])
        
        self.dropout = nn.Dropout(dropoutfreq)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc_in(x)))
        x = self.dropout(x)
        for i in range(self.nlayers):
            x = F.relu(self.bns[i]( self.fcs[i](x) ))
            x = self.dropout(x)
        x = self.fc_out(x)
        if self.classification:
            x = self.out_activation(x)
        return x
