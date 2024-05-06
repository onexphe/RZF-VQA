'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-04-12 17:22:00
LastEditors: onexph
LastEditTime: 2022-04-12 18:31:05
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model.fc as FC
from torch.autograd import Variable
from .attention import SanAttention, apply_attention
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
import pdb
from utils import freeze_layer


class GloveFact(nn.Module):
    def __init__(self, args,embedding_weights=None,rnn_bidirectional=True):
        super(GloveFact, self).__init__()

        question_features = 1024
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)

        self.w_emb = WordEmbedding(embedding_weights.size(0), 300, .0)
        if args.freeze_w2v:
            self.w_emb.init_embedding(embedding_weights)
            freeze_layer(self.w_emb)

        self.drop = nn.Dropout(0.5)
        self.text = Seq2SeqRNN(
            input_features=embedding_weights.size(1),
            rnn_features=int(rnn_features),
            rnn_type='LSTM',
            rnn_bidirectional=rnn_bidirectional,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, q, q_len):
        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))

        return q