'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-04-19 11:27:06
LastEditors: onexph
LastEditTime: 2022-04-25 16:03:54
'''
'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-04-19 11:27:06
LastEditors: onexph
LastEditTime: 2022-04-19 16:45:53
'''
# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .lxrt.entry import LXRTEncoder
from .lxrt.modeling import BertLayerNorm, GeLU
from .fc import GroupMLP

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class LXMERT(nn.Module):
    def __init__(self,args,dataset,_):
        super().__init__()
        self.args = args
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            self.args.LXMERT,
            max_seq_length=MAX_VQA_LENGTH,
            mode='xlr'
        )
        hid_dim = self.lxrt_encoder.dim

        if self.args.fact_map:
            self.output = 0
        elif self.args.relation_map:
            self.output = 1
        else:
            self.output = 2

        # self.gmlp_v = GroupMLP(
        #     in_features=hid_dim*36,
        #     mid_features= 4 * args.embedding_size,#args.hidden_size,
        #     out_features=args.embedding_size,
        #     drop=0.5,
        #     groups=64,
        # )
        # self.gmlp_l = GroupMLP(
        #     in_features=hid_dim*20,
        #     mid_features= 4 * args.embedding_size,#args.hidden_size,
        #     out_features=args.embedding_size,
        #     drop=0.5,
        #     groups=64,
        # )
        # self.gmlp_x = GroupMLP(
        #     in_features=hid_dim,
        #     mid_features= 2048,
        #     out_features=args.embedding_size,
        #     drop=0.5,
        #     groups=64,
        # )
        # self.mlp = nn.Linear(hid_dim*56, 1024)  
#         for m in self.modules():
#             if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#                 init.xavier_uniform(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
        self.lstm = nn.LSTM(hid_dim ,hid_dim, 2,bidirectional=True, batch_first=True, dropout=0.1)
        self.dropout_r = nn.Dropout(0.1)
        self.fc_rnn = nn.Linear(hid_dim * 2, 1024)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, hid_dim)) for k in (2,3,4)])
        self.dropout_c = nn.Dropout(0.1)
        self.fc_cnn = nn.Linear(256 * 3, 1024)

        self.fc = nn.Linear(1024*2, 1024)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        (lang_feats, visn_feats), pooled_output = self.lxrt_encoder(sent, (feat, pos))
        if self.output == 0:
            x = visn_feats #[128, 36, 768]
            x = lang_feats
        elif self.output == 1:
            x = lang_feats  #[128, 20, 768]
        else:
            # x = pooled_output   #[128,768]

            out_r, (hidden, cell)= self.lstm(lang_feats)
            out_r = self.dropout_r(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            out_r = self.fc_rnn(out_r.squeeze(0))

            # out_c = visn_feats.unsqueeze(1)
            # out_c = torch.cat([self.conv_and_pool(out_c, conv) for conv in self.convs], 1)
            # out_c = self.dropout_c(out_c)
            # out_c = self.fc_cnn(out_c)

            # print(out_c.shape,out_r.shape)
            # out_c = self.fc(torch.cat((out_r, out_c), dim=1))
            return out_r



        return x