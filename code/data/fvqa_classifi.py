'''
Description: 
Version: 1.0
Autor: onexph
Date: 2022-03-29 16:25:51
LastEditors: onexph
LastEditTime: 2022-03-29 17:09:26
'''
import json
import os
import os.path as osp
import nltk
from collections import Counter
import torch
import torch.utils.data as data
import pdb

################
from .base import VisualQA
from .preprocess import process_punctuation


def get_loader(args, vector, train=False, val=False):
    """ Returns a data loader for the desired split """
    assert train + val == 1, 'need to set exactly one of {train, val, test} to True'  # 必须有一个为真
    id = args.FVQA.data_choice
    if train:
        filepath = "train" + id
        print("use train data:", id)
        filepath = os.path.join(args.FVQA.train_data_path, filepath)
    else:
        filepath = "test" + id
        filepath = os.path.join(args.FVQA.test_data_path, filepath)

    split = FVQA(  # 定义每一次训练的VQA输入 # ok
        args,
        path_for(args, train=train, val=val, filepath=filepath),  # train的问题
        vector,  # 对应的词向量
        file_path=filepath
    )
    batch_size = args.TRAIN.batch_size
    if val:
        batch_size = args.TEST.batch_size
    print("batch_size",batch_size)
    loader = torch.utils.data.DataLoader(  # 定义传统的DataLoader
        split,
        batch_size=batch_size,
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=args.TRAIN.data_workers,
    )

    return loader

class FVQA(VisualQA):  # ok
    """ FVQA dataset, open-ended """

    def __init__(self, args, qa_path, vector, file_path=None):
        self.args = args
        super(FVQA, self).__init__(args, vector)
        # load annotation
        with open(qa_path, 'r') as fd:
            self.qa_json = json.load(fd)

        # print('extracting answers...')

        # 把问题变成id向量+长度的表示, 答案变成id向量
        self.questions = list(prepare_questions(self.qa_json,self.answer_to_index))  # 候选答案列表的列表



    def __getitem__(self, item):  # ok

        question, label = self.questions[item]  # 问题向量列表

        return  question, label
    


def path_for(args, train=False, val=False, filepath=""):
    # tra = "all_qs_dict_release_train_" + str(args.FVQA.max_ans) + ".json"
    # tes = "all_qs_dict_release_test_" + str(args.FVQA.max_ans) + ".json"
    tra = "all_qs_dict_release_train_500.json"
    tes = "all_qs_dict_release_test_500.json"
    if train == True:
        return os.path.join(args.FVQA.train_data_path, filepath, tra)
    else:
        return os.path.join(args.FVQA.test_data_path, filepath, tes)


def prepare_questions(questions_json,answer2index):  # ok
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    keys = list(questions_json.keys())
    questions = []
    for a in keys:
        question  = questions_json[a]['question']
        question = question.lower()[:-1]
        relation = questions_json[a]['fact'][1]
        index = answer2index[relation]


        questions.append([question,index])  # question的list
    return questions