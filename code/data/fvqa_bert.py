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
from model.re_agcn.tokenization import BertTokenizer 
from model.re_agcn.bert import BertPreTrainedModel, BertModel ,BertConfig


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
        answer_vocab_path = self.args.FVQA.answer_vocab_path
        super(FVQA, self).__init__(args, vector)
        # load annotation
        with open(qa_path, 'r') as fd:
            self.qa_json = json.load(fd)

        # print('extracting answers...')

        # 把问题变成id向量+长度的表示, 答案变成id向量
        if args.fact_map:
            #  得到对应的名字
            name = "fact"
            self.answers = list(prepare_fact(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示
        elif args.relation_map:
            name = "relation"
            self.answers = list(prepare_relation(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示
        else:
            name = "answer"
            self.answers = list(prepare_answers(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示

        cache_filepath = self._get_cache_path(qa_path, file_path, name)

        self.questions,self.questions_vec ,self.answer_indices = self._qa_id_represent(cache_filepath)

        self.max_seq_length = 30
        self.tokenizer = BertTokenizer('./model/re_agcn/vocab.txt')

    def open_hdf5(self):
        self.image_features_path = self.args.FVQA.feature_path
        self.image_id_to_index = self._create_image_id_to_index()  # 得到图片编号到下标的表示
        self.image_ids = self._get_img_id()

    def __getitem__(self, item):  # ok
        if not hasattr(self, 'image_ids'):
            self.open_hdf5()
        question_vec, question_length = self.questions_vec[item]  # 问题向量列表
        question = self.questions[item]
        label = self._encode_multihot_labels(self.answers[item])  # 答案的multihot表示 前百分之多少的答案
        # image_id = self.image_ids[item]
        # image, spa = self._load_image(image_id)  # 直接获得图片的特征

        #bert
        tokens = []
        token = self.tokenizer.tokenize(question)
        tokens.extend(token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        return  question, label, item, question_length,input_ids, segment_ids, attention_mask

    def _get_cache_path(self, qa_path, file_path, name):
        w2v = ""
        if "KG" in self.args.method_choice:
            if "w2v" in self.args.FVQA.entity_path:
                w2v = "(w2vinit)_" + self.args.FVQA.entity_num + "_" + self.args.FVQA.KGE
            else:
                w2v = "_" + self.args.FVQA.entity_num + "_" + self.args.FVQA.KGE
        if "train" in qa_path:
            cache_filepath = osp.join(file_path, "fvqa_" + name + "_and_question_train_" +
                                      self.args.method_choice + w2v + "_" + str(self.args.FVQA.max_ans) + ".pt")
        else:
            cache_filepath = osp.join(file_path, "fvqa_" + name + "_and_question_test_" + self.args.method_choice + w2v + "_" + str(
                self.args.FVQA.max_ans) + ".pt")
        return cache_filepath

    def _qa_id_represent(self, cache_filepath):
        if not os.path.exists(cache_filepath):
            # print('encoding questions...')
            questions = list(prepare_questions(self.qa_json))  # 问题词列表的列表
            questions_vec = [self._encode_question(q) for q in questions]  # 把问题变成id向量+长度的表示

            # 对于候选答案列表中的每一个问题对应的候选答案列表，转换成下标表示[[1,2,3],[2,3,4]......]  1——>一个答案
            answer_indices = [[self.answer_to_index.get(_a, -1) for _a in a] for a in self.answers]  # 如果没有匹配就是 -1

            questions = prepare_questions_text(self.qa_json)

            torch.save({'questions': questions, 'questions_vec': questions_vec,'answer_indices': answer_indices}, cache_filepath)

        else:
            # 已经有，对应这个训练/测试集 的问题w2v表，[train 和 test是不一样的]
            _cache = torch.load(cache_filepath)
            questions = _cache['questions']  # 词向量列表 + 长度
            questions_vec = _cache['questions_vec']
            answer_indices = _cache['answer_indices']  # 答案下标
            # self.answer_vectors = _cache['answer_vectors']  # 答案的向量表示[平均]

        return questions,questions_vec, answer_indices

    def _get_img_id(self):
        image_ids = []
        keys = list(self.qa_json.keys())
        for a in keys:
            filename = self.qa_json[a]["img_file"]
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            if not filename.endswith('.jpg'):
                id += 1000000  # 把jpg和jpeg的分开
                # pdb.set_trace()
            image_ids.append(id)
        return image_ids

def path_for(args, train=False, val=False, filepath=""):
    # tra = "all_qs_dict_release_train_" + str(args.FVQA.max_ans) + ".json"
    # tes = "all_qs_dict_release_test_" + str(args.FVQA.max_ans) + ".json"
    tra = "all_qs_dict_release_train_500.json"
    tes = "all_qs_dict_release_test_500.json"
    if train == True:
        return os.path.join(args.FVQA.train_data_path, filepath, tra)
    else:
        return os.path.join(args.FVQA.test_data_path, filepath, tes)


def prepare_questions(questions_json):  # ok
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    keys = list(questions_json.keys())
    questions = []
    for a in keys:
        questions.append(questions_json[a]['question'])  # question的list
    for question in questions:
        question = question.lower()[:-1]
        yield nltk.word_tokenize(process_punctuation(question))  # 得到一个词的list，例如['I', 'LOVE', 'YOU']

def prepare_questions_text(questions_json):
    keys = list(questions_json.keys())
    questions = []
    for a in keys:
        questions.append(questions_json[a]['question'])  # question的list

    return questions

def prepare_answers(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    answers = []

    for a in keys:
        answer = answers_json[a]["answer"]
        answers.append([answer] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for answer_list in answers:
        ret = list(map(process_punctuation, answer_list))  # 去除标点等操作
        yield ret


def prepare_fact(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    support_facts = []
    for a in keys:
        answer = answers_json[a]["answer"]
        facts = answers_json[a]["fact"]
        f1 = facts[0]
        f2 = facts[2]
        if answer != f1 and answer != f2:
            pdb.set_trace()
        assert (answer == f1 or answer == f2)
        if answer == f1:
            fact = f2
        else:
            fact = f1
        support_facts.append([fact] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for support_facts_list in support_facts:
        ret = list(map(process_punctuation, support_facts_list))  # 去除标点等操作
        yield ret


def prepare_relation(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    relations = []
    for a in keys:
        facts = answers_json[a]["fact"]
        relation = facts[1]

        relations.append([relation] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for relation_list in relations:
        ret = list(map(process_punctuation, relation_list))  # 去除标点等操作
        yield ret

def prepare_entity(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    support_facts = []
    for a in keys:
        answer = answers_json[a]["answer"]
        facts = answers_json[a]["fact"]
        f1 = facts[0]
        f2 = facts[2]
        if answer != f1 and answer != f2:
            pdb.set_trace()
        assert (answer == f1 or answer == f2)
        support_facts.append([f1] * 10)  # 双层list，内层的list对应一个问题的答案序列
        support_facts.append([f2] * 10)
    for support_facts_list in support_facts:
        ret = list(map(process_punctuation, support_facts_list))  # 去除标点等操作
        yield ret