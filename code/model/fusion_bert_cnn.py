'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-03-27 23:08:46
LastEditors: onexph
LastEditTime: 2022-04-19 16:48:16
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .attention import SanAttention, apply_attention
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
from model import Vector, SimpleClassifier
import pdb
import os 

from .re_agcn.bert import BertPreTrainedModel, BertModel ,BertConfig

class BERT(BertPreTrainedModel):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args,config,dataset,embedding_weights=None,rnn_bidirectional=True):
        super(BERT, self).__init__(config)
        question_features = 1024
        vision_features = args.output_features
        bertOut = 768
        # self.bert = BertModel(config)
        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config = config)

        for param in self.bert.parameters():
            param.requires_grad = False
        # for param in self.bert.pooler.parameters():
        #     param.requires_grad = True

        for name, p in self.bert.named_parameters():
            print(f'{name}:\t{p.requires_grad}')

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (2,3,4)])
        self.dropout = nn.Dropout(0.1)

        self.fc_cnn = nn.Linear(256 * 3, 1024)

        self.mlp = GroupMLP(
            in_features=bertOut,
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,  q, q_len,input_ids, segment_ids, attention_mask):
        # pdb.set_trace()

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask, output_all_encoded_layers=False)


        # combined = torch.cat([v, q], dim=1)
        # sequence_output = sequence_output.view(sequence_output.shape[0],-1)
        # embedding = self.mlp(pooled_output)

        out = sequence_output.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)

        return  out#embedding#pooled_output

