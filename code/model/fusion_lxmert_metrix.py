'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-04-19 11:27:06
LastEditors: onexph
LastEditTime: 2022-04-27 22:15:30
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
from .language_model import Seq2SeqRNN, WordEmbedding

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

        self.gmlp_v = GroupMLP(
            in_features=hid_dim*36,
            mid_features= 4 * args.embedding_size,#args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        self.gmlp_l = GroupMLP(
            in_features=hid_dim*20,
            mid_features= 4 * args.embedding_size,#args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        self.gmlp_x = GroupMLP(
            in_features=hid_dim,
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )

        self.v_att_proj = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        self.linear_1024 = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 1024))
        self.l48to24 = nn.Linear(2048+1024, 1024)

        # question_features = 1024
        # rnn_features = int(question_features // 2)
        # self.text = Seq2SeqRNN(
        #     input_features=embedding_weights.size(1),
        #     rnn_features=int(rnn_features),
        #     rnn_type='LSTM',
        #     rnn_bidirectional=True,
        # )

        # self.mlp = nn.Linear(hid_dim*56, 1024)  
#         for m in self.modules():
#             if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#                 init.xavier_uniform(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()

    def forward(self,allFeat, feat, pos, sent,q,q_len):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        (lang_feats, visn_feats), pooled_output = self.lxrt_encoder(sent, (feat, pos))
        if self.output == 3:
            
            # q = F.normalize(self.text(q, list(q_len.data)), p=2, dim=1)  # 问题向量求平均值
            v = allFeat
            v = F.normalize(F.avg_pool2d(v, (v.size(2), v.size(3))).squeeze(), p=2, dim=1)

            sim_matrix_v2l = torch.matmul(visn_feats, lang_feats.transpose(1,2))  # b * v_length * l_length
            kg_output, k = torch.topk(sim_matrix_v2l, dim=-1, k=1)

            # hard attention
            # hard_attention_value = gumbel_softmax(kg_output.squeeze())
            # head = (visn_feats * hard_attention_value.unsqueeze(-1)).sum(-2)

            # soft attention
            kg_output = F.softmax(kg_output.squeeze(), dim=-1)
            head = (visn_feats * kg_output.unsqueeze(-1)).sum(-2)

            x = self.linear_1024(head)
            x = torch.cat([v, x], dim=1)
#             print(x.shape)
            x = self.l48to24(x)
    
        elif self.output == 0:
            x = lang_feats
        elif self.output == 1:
            x = lang_feats  #[128, 20, 768]
        else:
            x = pooled_output   #[128,768]

        x = x.view(x.shape[0],-1)
        x = self.gmlp_l(x)
        return x


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard