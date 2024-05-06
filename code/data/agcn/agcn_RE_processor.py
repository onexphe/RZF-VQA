import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .dep_parser import DepInstanceParser


def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

class RE_Processor():
    def __init__(self, direct=True, dep_type="first_order", types_dict={}, labels_dict={}):
        self.direct = direct
        self.dep_type = dep_type
        self.types_dict = types_dict
        self.labels_dict = labels_dict

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="train"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="dev"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="test"), "test")

    def get_knowledge_feature(self, data_dir, flag="train"):
        return self.read_features(data_dir, flag=flag)

    def get_labels(self, data_dir):
        label_path = os.path.join(data_dir, "label.json")
        with open(label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def get_dep_labels(self, data_dir):
        dep_labels = ["self_loop"]
        dep_type_path = os.path.join(data_dir, "dep_type.json")
        with open(dep_type_path, 'r') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if self.direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        return dep_labels

    def get_key_list(self):
        return self.keys_dict.keys()

    def _create_examples(self, features, set_type):
        examples = []
        for i, feature in enumerate(features):
            guid = "%s-%s" % (set_type, i)
            feature["guid"] = guid
            examples.append(feature)
        return examples

    def prepare_keys_dict(self, data_dir):
        keys_frequency_dict = defaultdict(int)
        for flag in ["train", "test", "dev"]:
            datafile = os.path.join(data_dir, '{}.txt'.format(flag))
            if os.path.exists(datafile) is False:
                continue
            all_data = self.load_textfile(datafile)
            for data in all_data:
                for word in data['words']:
                    keys_frequency_dict[change_word(word)] += 1
        keys_dict = {"[UNK]":0}
        for key, freq in sorted(keys_frequency_dict.items(), key=lambda x: x[1], reverse=True):
            keys_dict[key] = len(keys_dict)
        self.keys_dict = keys_dict
        self.keys_frequency_dict = keys_frequency_dict
        print(keys_dict)

    def prepare_type_dict(self, data_dir):
        dep_type_list = self.get_dep_labels(data_dir)
        types_dict = {"none": 0}
        for dep_type in dep_type_list:
            types_dict[dep_type] = len(types_dict)
        self.types_dict = types_dict
        print(types_dict)

    def prepare_labels_dict(self, data_dir):
        label_list = self.get_labels(data_dir)
        labels_dict = {}
        for label in label_list:
            labels_dict[label] = len(labels_dict)
        self.labels_dict = labels_dict
        print(labels_dict)

    def read_features(self, data_dir, flag):

        type = 'train'
        if not os.path.exists(os.path.join(data_dir,  'dep_train_500.json')):
            type = 'test'
        all_dep_info = self.load_depfile(os.path.join(data_dir,  'dep_%s_500.json'%(type)))
        all_text_data = self.load_textfile(os.path.join(data_dir,  'all_qs_dict_release_%s_500.json'%(type)))
        all_feature_data = []
        for text_data,dep_info in zip(all_text_data, all_dep_info):
            tokens = text_data["words"]
            dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=tokens)
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order(direct=self.direct)
            all_feature_data.append({
                    "words": dep_instance_parser.words,
                    "dep_adj_matrix": dep_adj_matrix,
                    "dep_type_matrix": dep_type_matrix,
                })

        return all_feature_data

    def read_features_(self, data_dir, flag):
        all_text_data = self.load_textfile(os.path.join(data_dir,  '{}.txt'.format(flag)))
        all_dep_info = self.load_depfile(os.path.join(data_dir,  '{}.txt.dep'.format(flag)))
        all_feature_data = []
        for text_data,dep_info in zip(all_text_data, all_dep_info):
            label = text_data["label"]
            if label == "other":
                label = "Other"

            ori_sentence = text_data["ori_sentence"].split(" ")
            tokens = text_data["words"]
            e11_p = ori_sentence.index("<e1>")  # the start position of entity1
            e12_p = ori_sentence.index("</e1>")  # the end position of entity1
            e21_p = ori_sentence.index("<e2>")  # the start position of entity2
            e22_p = ori_sentence.index("</e2>")  # the end position of entity2

            if e11_p < e21_p:
                start_range = list(range(e11_p, e12_p - 1))
                end_range = list(range(e21_p - 2, e22_p - 3))
            else:
                start_range = list(range(e11_p - 2, e12_p - 3))
                end_range = list(range(e21_p, e22_p - 1))

            dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=tokens)
            if self.dep_type == "first_order" or self.dep_type == "full_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order(direct=self.direct)
            elif self.dep_type == "local_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_local_graph(start_range, end_range, direct=self.direct)
            elif self.dep_type == "global_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_global_graph(start_range, end_range, direct=self.direct)
            elif self.dep_type == "local_global_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_local_global_graph(start_range, end_range, direct=self.direct)

            all_feature_data.append({
                "words": dep_instance_parser.words,
                "ori_sentence": ori_sentence,
                "dep_adj_matrix": dep_adj_matrix,
                "dep_type_matrix": dep_type_matrix,
                "label": label,
                "e1":text_data["e1"],
                "e2":text_data["e2"],
            })

        return all_feature_data

    def load_depfile_(self, filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data

    def load_depfile(self, filename):
        data = []
        with open(filename,'r')as f:
            dep_data = json.load(f)
            for key,value in dep_data.items():
                dep_info = []
                for one in value:
                    dep_info.append({
                        "governor": int(one[0]),
                        "dependent": int(one[1]),
                        "dep": one[2],
                    })
                data.append(dep_info)
        return data

    def load_textfile_(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split("\t")
                if len(items) != 4:
                    continue
                e1,e2,label,sentence = items
                data.append({
                    "e1":e1,
                    "e2":e2,
                    "label":label,
                    "ori_sentence":sentence,
                    "words": [token for token in sentence.split(" ") if token not in ["<e1>", "</e1>", "<e2>", "</e2>"]]
                })
        return data

    def load_textfile(self, filename):
        data = []
        with open(filename,'r')as f:
            text_data = json.load(f)
            for key,value in text_data.items():
                sentence = value['question']
                data.append({
                    "ori_sentence":sentence,
                    "words": [token for token in sentence.split(" ") if token not in ["<e1>", "</e1>", "<e2>", "</e2>"]]
                })
        return data


    def convert_examples_to_features_(self, examples, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = self.labels_dict
        dep_label_map = self.types_dict

        features = []
        b_use_valid_filter = False
        for (ex_index, example) in enumerate(examples):
            tokens = ["[CLS]"]
            valid = [0]
            e1_mask = []
            e2_mask = []
            e1_mask_val = 0
            e2_mask_val = 0
            entity_start_mark_position = [0, 0]
            for i, word in enumerate(example["ori_sentence"]):
                if len(tokens) >= max_seq_length - 1:
                    break
                if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                    tokens.append(word)
                    valid.append(0)
                    if word in ["<e1>"]:
                        e1_mask_val = 1
                        entity_start_mark_position[0] = len(tokens) - 1
                    elif word in ["</e1>"]:
                        e1_mask_val = 0
                    if word in ["<e2>"]:
                        e2_mask_val = 1
                        entity_start_mark_position[1] = len(tokens) - 1
                    elif word in ["</e2>"]:
                        e2_mask_val = 0
                    continue

                token = tokenizer.tokenize(word)
                if len(tokens) + len(token) > max_seq_length - 1:
                    break
                tokens.extend(token)
                e1_mask.append(e1_mask_val)
                e2_mask.append(e2_mask_val)
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
                        b_use_valid_filter = True

            tokens.append("[SEP]")
            valid.append(0)
            e1_mask.append(0)
            e2_mask.append(0)
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            valid += padding
            e1_mask += [0] * (max_seq_length - len(e1_mask))
            e2_mask += [0] * (max_seq_length - len(e2_mask))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(e1_mask) == max_seq_length
            assert len(e2_mask) == max_seq_length

            max_words_num = sum(valid)
            def get_adj_with_value_matrix(dep_adj_matrix, dep_type_matrix):
                final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                for pi in range(max_words_num):
                    for pj in range(max_words_num):
                        if dep_adj_matrix[pi][pj] == 0:
                            continue
                        if pi >= max_seq_length or pj >= max_seq_length:
                            continue
                        final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                        final_dep_type_matrix[pi][pj] = dep_label_map[dep_type_matrix[pi][pj]]
                return final_dep_adj_matrix, final_dep_type_matrix

            dep_adj_matrix, dep_type_matrix = get_adj_with_value_matrix(example["dep_adj_matrix"], example["dep_type_matrix"])

            label_id = label_map[example["label"]]

            if ex_index < 5:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example["guid"]))
                logging.info("sentence: %s" % (example["ori_sentence"]))
                logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info("valid: %s" % " ".join([str(x) for x in valid]))
                logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logging.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
                logging.info("dep_adj_matrix: %s" % " ".join([str(x) for x in dep_adj_matrix]))
                logging.info("dep_type_matrix: %s" % " ".join([str(x) for x in dep_type_matrix]))
                logging.info("label: %s (id = %d)" % (example["label"], label_id))

            features.append({
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_id": label_id,
                "valid_ids": valid,
                "e1_mask": e1_mask,
                "e2_mask": e2_mask,
                "dep_adj_matrix": dep_adj_matrix,
                "dep_type_matrix": dep_type_matrix,
                "b_use_valid_filter": b_use_valid_filter,
                "entity_start_mark_position":entity_start_mark_position
            })
        return features
    def convert_examples_to_features(self, examples, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        dep_label_map = self.types_dict

        features = []
        for (ex_index, example) in enumerate(examples):
            max_words_num = 30
            def get_adj_with_value_matrix(dep_adj_matrix, dep_type_matrix):
                final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                for pi in range(max_words_num):
                    for pj in range(max_words_num):
                        if pi >= max_seq_length or pj >= max_seq_length or pi >= len(dep_adj_matrix) or pj >= len(dep_adj_matrix) :
                            continue
                        if  dep_adj_matrix[pi][pj] == 0:
                            continue
                        final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                        final_dep_type_matrix[pi][pj] = dep_label_map[dep_type_matrix[pi][pj]]
                return final_dep_adj_matrix, final_dep_type_matrix

            dep_adj_matrix, dep_type_matrix = get_adj_with_value_matrix(example["dep_adj_matrix"], example["dep_type_matrix"])

            features.append({
                "dep_adj_matrix": dep_adj_matrix,
                "dep_type_matrix": dep_type_matrix,
            })
        return features

    # def build_dataset(self, examples, tokenizer, max_seq_length, mode, args):
    #     features = self.convert_examples_to_features(examples, tokenizer, max_seq_length)
    #     if args.local_rank != -1 and mode == "train":
    #         features = features[args.rank::args.world_size]
    #     return REDataset(features, max_seq_length)
