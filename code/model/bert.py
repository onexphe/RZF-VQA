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
import pdb
import os 

from .re_agcn.bert import BertPreTrainedModel, BertModel ,BertConfig

class BERT(BertPreTrainedModel):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args,config):
        super(BERT, self).__init__(config)
        question_features = 1024
        bertOut = 768
        # self.bert = BertModel(config)
        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config = config)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.pooler.parameters():
            param.requires_grad = False

        for name, p in self.bert.named_parameters():
            print(f'{name}:\t{p.requires_grad}')

        # self.apply(self.init_bert_weights)

        self.mlp = GroupMLP(
            in_features=bertOut,
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )


    def forward(self,input_ids, segment_ids, attention_mask):
        # pdb.set_trace()

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask, output_all_encoded_layers=False)


        # combined = torch.cat([v, q], dim=1)
        # sequence_output = sequence_output.view(sequence_output.shape[0],-1)
        # embedding = self.mlp(pooled_output)
        return pooled_output# embedding# sequence_output.view(sequence_output.shape[0],-1)#,pooled_output