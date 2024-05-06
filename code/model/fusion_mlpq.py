import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .fc import GroupMLP
from .language_model import WordEmbedding,Seq2SeqRNN


class MLPQ(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    # def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset, embedding_weights=None, rnn_bidirectional=True):
        super(MLPQ, self).__init__()
        embedding_requires_grad = not args.freeze_w2v  # freeze 则不需要grad
        
        question_features = 1024
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)

        self.mlp = GroupMLP(
            in_features=question_features,
            mid_features=args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )
        
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

    def forward(self, v, b, q, q_len):

        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))

        embedding = self.mlp(q)
        return embedding


class BagOfWordsProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features,
                 embedding_weights, embedding_requires_grad):
        super(BagOfWordsProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.embedding.weight.data = embedding_weights
        self.embedding.weight.requires_grad = embedding_requires_grad

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        q_len = Variable(torch.Tensor(q_len).view(-1, 1) + 1e-12, requires_grad=False).cuda()

        return torch.div(torch.sum(embedded, 1), q_len)
