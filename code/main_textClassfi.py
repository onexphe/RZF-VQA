'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-03-29 17:09:45
LastEditors: onexph
LastEditTime: 2022-03-29 17:24:10
'''
import os,json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import fvqa_classifi
from config import cfg
from torchlight import initialize_exp, set_seed, snapshot, get_dump_path, show_params
from utils import unseen_mask, freeze_layer, cosine_sim, Metrics, instance_bce_with_logits
from model import Vector, SimpleClassifier

class Runner:
    def __init__(self, args):
        self.word2vec = Vector(args.FVQA.common_data_path)
        self.train_loader = fvqa_classifi.get_loader(args, self.word2vec, train=True)
        self.val_loader = fvqa_classifi.get_loader(args, self.word2vec, val=True)
        
        self.train_data = json.load(open("/ws/code/text-classifi/question_relation_train.json", encoding="utf8"))
        self.test_data = json.load(open("/ws/code/text-classifi/question_relation_test.json", encoding="utf8"))
        pass
    def run(self):
        # data load
        tq = tqdm(self.train_loader, ncols=0)

        for question, answer in tq:
            for one in question:
                self.train_data['data'].append(one)
            for one in answer:
                self.train_data['target'].append(one.item())
            print(question,answer)
        tq = tqdm(self.val_loader, ncols=0)

        for question, answer in tq:
            for one in question:
                self.test_data['data'].append(one)
            for one in answer:
                self.test_data['target'].append(one.item())
        with open("/ws/code/text-classifi/question_relation_train.json","w") as dump_f:
            json.dump(self.train_data,dump_f)
        with open("/ws/code/text-classifi/question_relation_test.json","w") as dump_f:
            json.dump(self.test_data,dump_f)


if __name__ == '__main__':
    # Config loading...
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    set_seed(cfg.random_seed)

    # Environment initialization...
    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)
    if not cfg.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    torch.cuda.set_device(cfg.gpu_id)

    # Run...
    runner = Runner(cfg)
    runner.run()