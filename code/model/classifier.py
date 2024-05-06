'''
Description: 
Version: 1.0
Autor: onexph
Date: 2021-09-22 16:16:41
LastEditors: onexph
LastEditTime: 2022-01-09 14:02:45
'''
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
    
class SimpleClassifier_2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier_2, self).__init__()
#         layers = [
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            # weight_norm(nn.Linear(hid_dim, hid_dim), dim=None),
            # nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
#         ]
#         self.main = nn.Sequential(*layers)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(in_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.d1 = nn.Dropout(dropout, inplace=True)
        self.d2 = nn.Dropout(dropout, inplace=True)
    def forward(self, x,y):
#         print(x.shape,y.shape)
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        y = F.relu(self.fc2(y))
        y = self.d1(y)
        x = x + y
        x = self.fc3(x)
        return x
    
    
class SimpleClassifier_initW(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier_initW, self).__init__()

        linear1 = nn.Linear(in_dim, hid_dim)
        # linear1.apply(self.init_weight)
        # torch.nn.init.xavier_uniform(linear1.weight)
        torch.nn.init.normal_(linear1.weight, mean=1, std=1)
        linear2 = nn.Linear(hid_dim, out_dim)
        # linear2.apply(self.init_weight)
        # torch.nn.init.xavier_uniform(linear2.weight)
        torch.nn.init.normal_(linear2.weight, mean=1, std=1)
        
        layers = [
            weight_norm(linear1,dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(linear2,dim=None)
        ]
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
    
    def init_weight(m):
        if type(m) == nn.Linear:
            # torch.nn.init.xavier_uniform(m.weight)
            torch.nn.init.normal_(m.weight, mean=1, std=0)
            m.bias.data.fill_(0.01)
