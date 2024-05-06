# -*- coding: utf-8 -*-
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss

from .re_agcn.bert import BertConfig, BertModel, BertPreTrainedModel


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight.float())
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj.float(), hidden) / denom
        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))

class TypeGraphConvolution(nn.Module):
    """
    Type GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(TypeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        val_sum = val_us + dep_embed
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(val_sum.float(), self.weight.float())
        output = hidden.transpose(1,2) * adj_us.float()
        output = torch.sum(output, dim=2)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))

class AGCN(BertPreTrainedModel):
    def __init__(self, args, config, dataset,embedding_weights=None):
        super(AGCN, self).__init__(config)

        # config = BertConfig.from_json_file(os.path.join("./model/re_agcn/bert_config.json"))
        config.__dict__["num_gcn_layers"] = 1
        config.__dict__["num_labels"] = 10
        config.__dict__["type_num"] = 82
        config.__dict__["dep_type"] = 'local_global_graph'

        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config = config)
        for param in self.bert.parameters():
            param.requires_grad = False
        for name, p in self.bert.named_parameters():
            print(f'{name}:\t{p.requires_grad}')


        self.dep_type_embedding = nn.Embedding(config.type_num, config.hidden_size, padding_idx=0)
        gcn_layer = TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])
        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm = nn.LSTM(768 ,768, 2,bidirectional=True, batch_first=True, dropout=0.1)
        self.fc_rnn = nn.Linear(768 * 2, 1024)


    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1)
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / feat_dim ** 0.5
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # sequence_output = self.dropout(sequence_output)

        dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
        dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
        for i, gcn_layer_module in enumerate(self.gcn_layer):
            attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix)
            sequence_output = gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)

        out, (hidden, cell)= self.lstm(sequence_output)
        out = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        out = self.fc_rnn(out.squeeze(0))
        # out = self.fc_rnn(torch.cat((out.squeeze(0),pooled_output),dim=1))
        return out


        # e1_h = self.extract_entity(sequence_output, e1_mask)
        # e2_h = self.extract_entity(sequence_output, e2_mask)
        # pooled_output = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        # pooled_output = self.dropout(pooled_output)
