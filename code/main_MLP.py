import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
import warnings
from pprint import pprint
from torch.nn import CrossEntropyLoss, MSELoss


# self-defined
import model.fusion_net as fusion_net
import model.answer_net as answer_net
import model.gcn as gcn
from model import Vector, SimpleClassifier_2
from config import cfg
from torchlight import initialize_exp, set_seed, snapshot, get_dump_path, show_params
from utils import unseen_mask, freeze_layer, cosine_sim, Metrics, instance_bce_with_logits
from data import fvqa
import json
from utils.tool import fact2entity_class
import copy

# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings('ignore')


class Runner:
    def __init__(self, args):

        self.log_dir = get_dump_path(args)
        self.word2vec = Vector(args.FVQA.common_data_path)

        self.train_loader = fvqa.get_loader(args, self.word2vec, train=True)
        self.val_loader = fvqa.get_loader(args, self.word2vec, val=True)
        self.avocab = default_collate(list(range(0, args.FVQA.max_ans)))
        self.avocab_fact = default_collate(list(range(0, 2791)))
        self.avocab_relation = default_collate(list(range(0, 103)))
        self.fact2entityClass = fact2entity_class(args)

        self.question_word2vec = self.word2vec._prepare(self.train_loader.dataset.token_to_index)

        # answer choice
        assert args.method_choice in ['CLS', 'W2V', 'KG', 'GAE', 'KG_W2V', 'KG_GAE', 'GAE_W2V', 'KG_GAE_W2V']
        ans_len_table = {'W2V': 300, 'KG': 300, 'GAE': 1024, 'CLS': 0}

        self.method_list = args.method_choice.split('_')
        self.method_list.sort()
        for i in self.method_list:
            args.ans_feature_len += ans_len_table[i]

        # get fusion_model and answer_net
        self.fusion_model_ans, self.answer_net_ans = self._model_choice(args)
        self.fusion_model_rel, self.answer_net_rel = self._model_choice(args)
        self.fusion_model_fact, self.answer_net_fact = self._model_choice(args)

        #MLP
        self.MLP = SimpleClassifier_2(500, 500, 500, 0.5).cuda()


        # optimizer
        self.optimizer = optim.Adam(self.MLP.parameters(),lr=1e-4)

        # get the mask from zsl
        self.negtive_mux = unseen_mask(args, self.val_loader)

        # Recorder
        self.max_acc = [0, 0, 0, 0]
        self.max_zsl_acc = [0, 0, 0, 0]
        self.best_epoch = 0
        self.correspond_loss = 1e20


        self.args = args
        
        #train
        self.early_stop = 0
        # warm up (ref: ramen)
        self.gradual_warmup_steps = [i * self.args.TRAIN.lr for i in torch.linspace(0.5, 2.0, 7)]
        self.lr_decay_epochs = range(14, 47, self.args.TRAIN.lr_decay_step)
        self.bestAcc = 0
        self.iter = 0
        self.temp = True

        print("begin train! ...")
        print("loading model  ...")
        self._load_model(self.fusion_model_ans, "fusion", "answer")
        self._load_model(self.answer_net_ans, "embedding", "answer")
        self._load_model(self.fusion_model_rel, "fusion", "relation")
        self._load_model(self.answer_net_rel, "embedding", "relation")
        self._load_model(self.fusion_model_fact, "fusion", "fact")
        self._load_model(self.answer_net_fact, "embedding", "fact")

    def run(self):
        tmp_args = self.args
        tmp_args.fact_map = 1
        self.train_loader_fact = fvqa.get_loader(tmp_args, self.word2vec, train=True)
        tmp_args.fact_map = 0
        tmp_args.relation_map = 1
        self.train_loader_relation = fvqa.get_loader(tmp_args, self.word2vec, train=True)

        # well, we recommend only use the parameter : 'CLS' or 'W2V'.
        # if you wanna use 'KG_W2V', 'KG_GAE', 'GAE_W2V', 'KG_GAE_W2V'... you could modify the following concate code
        assert len(self.method_list) == 1

        for method_choice in self.method_list:
            answer_var, _ = self.train_loader.dataset._get_answer_vectors(method_choice, self.avocab)
            answer_var_fact, _ = self.train_loader_fact.dataset._get_answer_vectors(method_choice, self.avocab_fact)
            answer_var_relation, _ = self.train_loader_relation.dataset._get_answer_vectors(method_choice, self.avocab_relation)
            answer_var = F.normalize(answer_var, p=2, dim=1)
            answer_var_fact = F.normalize(answer_var_fact, p=2, dim=1)
            answer_var_relation = F.normalize(answer_var_relation, p=2, dim=1)

            # TODO: concate

        self.answer_var = Variable(answer_var.float()).cuda()
        self.answer_var_fact = Variable(answer_var_fact.float()).cuda()
        self.answer_var_relation = Variable(answer_var_relation.float()).cuda()


        self.val_metrics = Metrics()
        self.train_metrics = Metrics()
        ###################

        # if test:
        if self.args.now_test:
            self.args.TRAIN.epochs = 2

        for epoch in range(self.args.TRAIN.epochs):
            self.early_stop += 1
            if self.args.patience < self.early_stop:
                # early stop
                break
            # warm up
            if epoch < len(self.gradual_warmup_steps):
                self.optimizer.param_groups[0]['lr'] = self.gradual_warmup_steps[epoch]
            elif epoch in self.lr_decay_epochs:
                self.optimizer.param_groups[0]['lr'] *= self.args.TRAIN.lr_decay_rate
            
            self.train_metrics = Metrics()
            self.val_metrics = Metrics()


            # train
            if not self.args.now_test:
                ######## begin training!! #######
                self.train(epoch)
                #################################
                lr = self.optimizer.param_groups[0]['lr']
                # recode:
                logger.info(
                    f'Train Epoch {epoch}: LOSS={self.train_metrics.total_loss: .5f}, lr={lr: .6f}, acc1={self.train_metrics.acc_1: .2f},acc3={self.train_metrics.acc_3: .2f},acc10={self.train_metrics.acc_10: .2f}')
            # eval
            if epoch % 1 == 0 and epoch > 0:
                ######## begin evaling!! #######
                self.eval(epoch)
                #################################
                logger.info('#################################################################################################################')
                logger.info(f'Test Epoch {epoch}: LOSS={self.val_metrics.total_loss: .5f}, acc1={self.val_metrics.acc_1: .2f}, acc3={self.val_metrics.acc_3: .2f}, acc10={self.val_metrics.acc_10: .2f}')
                logger.info('#################################################################################################################')

                if self.val_metrics.acc_1 > (self.max_acc[0] + 0.2):
                    # reset early_stop and updata
                    self.early_stop = 0
                    self.best_epoch = epoch
                    self.correspond_loss = self.val_metrics.total_loss
                    self._updata_best_result(self.max_acc, self.val_metrics)

                    self.best_gcn_model = copy.deepcopy(self.MLP)


                if not args.no_tensorboard and not self.args.now_test:
                    writer.add_scalar('loss', self.val_metrics.total_loss, epoch)
                    writer.add_scalar('acc1', self.val_metrics.acc_1, epoch)
                    writer.add_scalar('acc3', self.val_metrics.acc_3, epoch)
                    writer.add_scalar('acc10', self.val_metrics.acc_10, epoch)      

        accNow = self.correctNum/self.allNum
        if accNow >= (self.bestAcc+0.005):
            # reset early_stop and updata
            self.bestAcc = accNow
            self.early_stop = 0
            self.best_epoch = epoch
            self.best_gcn_model = copy.deepcopy(self.MLP)
        logger.info(f'#### Eval top1={accNow},bestTop1={self.bestAcc},origin={self.correctNum_direct/self.allNum}')

        # save the model              
        if not self.args.now_test and self.args.save_model:
            self.gcn_model_path = self._save_model(self.best_gcn_model, "gcn")


    def train(self,epoch):

        self.get_min_and_max_flag = 0

        self.fusion_model_ans.eval()
        self.answer_net_ans.eval()
        self.fusion_model_rel.eval()
        self.answer_net_rel.eval()
        self.fusion_model_fact.eval()
        self.answer_net_fact.eval()

        self.MLP.train()

        tq = tqdm(self.train_loader)
        fact_relation_to_fact = self._get_fact_relation_dict()

        self.min = 1000
        self.max = 0

        for visual_features, boxes, question_features, answers, idx, q_len in tq:
            with torch.no_grad():
                visual_features = Variable(visual_features.float()).cuda()
                boxes = Variable(boxes.float()).cuda()
                question_features = Variable(question_features).cuda()
                answers = Variable(answers).cuda()
                q_len = Variable(q_len).cuda()

                fusion_embedading_ans = self.fusion_model_ans(visual_features, boxes, question_features, q_len)
                answer_embedding_ans = self.answer_net_ans(self.answer_var)

                fusion_embedading_rel = self.fusion_model_rel(visual_features, boxes, question_features, q_len)
                answer_embedding_rel = self.answer_net_rel(self.answer_var_relation)

                fusion_embedading_fact = self.fusion_model_fact(visual_features, boxes, question_features, q_len)
                answer_embedding_fact = self.answer_net_fact(self.answer_var_fact)

                predicts_rel = cosine_sim(fusion_embedading_rel,
                                          answer_embedding_rel) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_rel = predicts_rel.to(torch.float64)

                predicts_fact = cosine_sim(fusion_embedading_fact,
                                           answer_embedding_fact) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_fact = predicts_fact.to(torch.float64)

                predicts_ans = cosine_sim(fusion_embedading_ans,
                                          answer_embedding_ans) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_ans = predicts_ans.to(torch.float64)


            # MLP
            labels = answers.cuda()
            x,y = self._get_input(predicts_rel, predicts_fact, predicts_ans,fact_relation_to_fact)#[128,1000]
            output = self.MLP(x,y)    #[128,500]


            if True:
                loss_fct = CrossEntropyLoss()
                target = labels.argmax(dim=1)
                loss = loss_fct(output, target)

                #loss_fct = MSELoss()
                #target2 = F.softmax(predicts_ans.float(),dim=1)
                #loss2 = loss_fct(output.view(-1), target2.view(-1))
                #a = 10000
                #loss = loss + a*loss2 
            else:
                loss_fct = MSELoss()
                target2 = F.softmax(predicts_ans.float(),dim=1)
                loss = loss_fct(output.view(-1), target2.view(-1))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.train_metrics.update_per_batch(loss, answers.data, output.data)
        self.train_metrics.update_per_epoch()

    def eval(self,epoch):
        self.get_min_and_max_flag = 0

        self.fusion_model_ans.eval()
        self.answer_net_ans.eval()
        self.fusion_model_rel.eval()
        self.answer_net_rel.eval()
        self.fusion_model_fact.eval()
        self.answer_net_fact.eval()

        self.MLP.eval()

        tq = tqdm(self.val_loader)
        fact_relation_to_fact = self._get_fact_relation_dict()

        self.min = 1000
        self.max = 0
        self.itr = 0

        self.correctNum = 0
        self.correctNum_direct = 0
        self.allNum = 0

        for visual_features, boxes, question_features, answers, idx, q_len in tq:
            with torch.no_grad():
                visual_features = Variable(visual_features.float()).cuda()
                boxes = Variable(boxes.float()).cuda()
                question_features = Variable(question_features).cuda()
                answers = Variable(answers).cuda()
                q_len = Variable(q_len).cuda()

                fusion_embedading_ans = self.fusion_model_ans(visual_features, boxes, question_features, q_len)
                answer_embedding_ans = self.answer_net_ans(self.answer_var)

                fusion_embedading_rel = self.fusion_model_rel(visual_features, boxes, question_features, q_len)
                answer_embedding_rel = self.answer_net_rel(self.answer_var_relation)

                fusion_embedading_fact = self.fusion_model_fact(visual_features, boxes, question_features, q_len)
                answer_embedding_fact = self.answer_net_fact(self.answer_var_fact)

                predicts_rel = cosine_sim(fusion_embedading_rel, answer_embedding_rel) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_rel = predicts_rel .to(torch.float64)

                predicts_fact = cosine_sim(fusion_embedading_fact, answer_embedding_fact) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_fact = predicts_fact .to(torch.float64)

                predicts_ans = cosine_sim(fusion_embedading_ans, answer_embedding_ans) / self.args.loss_temperature  # temperature 与每一个答案都有计算相似度，
                predicts_ans = predicts_ans .to(torch.float64)


                # MLP
                labels = answers.cuda()
                x,y = self._get_input(predicts_rel, predicts_fact, predicts_ans,fact_relation_to_fact)#[128,1000]
                output = self.MLP(x,y)    #[128,500]

                if True:
                    loss_fct = CrossEntropyLoss()
                    target = labels.argmax(dim=1)
                    loss = loss_fct(output, target)

                # for i in range(labels.shape[0]):
                #     predict_lable = output[i].argmax()
                #     target = labels[i].argmax()
                #     oring_output = predicts_ans[i].argmax()

                #     self.iter += 1
                #     if self.iter% 100 == 0:
                #         logger.info(f'acc:{oring_output,predict_lable,target}')
                #         logger.info(f'loss:{loss}')
                #     if oring_output == target:
                #         self.correctNum_direct += 1
                #     if predict_lable == target:
                #         self.correctNum += 1
                #     self.allNum += 1

            self.val_metrics.update_per_batch(loss, answers.data, output.data)
        self.val_metrics.update_per_epoch()
        



        # if epoch==0:
        #     self.best_gcn_model = copy.deepcopy(self.MLP)
        # accNow = self.correctNum/self.allNum
        # if accNow >= (self.bestAcc+0.005):
        #     # reset early_stop and updata
        #     self.bestAcc = accNow
        #     self.early_stop = 0
        #     self.best_epoch = epoch
        #     self.best_gcn_model = copy.deepcopy(self.MLP)
        # logger.info(f'#### Eval top1={accNow},bestTop1={self.bestAcc},origin={self.correctNum_direct/self.allNum}')





    def _get_graph(self, predicts_rel, predicts_fact, predicts_ans,fact_relation_to_fact):
        #  ([128, 500])([128, 103])([128, 2791])
        mask_ans = torch.ones(predicts_rel.shape[0], self.args.FVQA.max_ans) # [128,500]

        # 把每一行的 predicts_rel 和 predicts_fact 中 top 拿出来凑对并且从fact_relation_to_ans筛选，将对应位置置0。rel top10，fact top30
        _, top_rel = predicts_rel.topk(self.args.top_rel, dim=1)
        _, top_fact = predicts_fact.topk(self.args.top_fact, dim=1)

        size = self.args.FVQA.max_ans + self.args.top_fact

        graph = torch.zeros(predicts_rel.shape[0], size,size) #[128,2790/500,500]
        adj = torch.zeros(predicts_rel.shape[0],size,size) #[128,2790/500,500]
        ##############################################################################
        # TODO: optimize this process via relation matric mapping rather than “ for-loop”
        for item in range(predicts_rel.shape[0]):
            for i in range(self.args.top_rel):
                for j in range(self.args.top_fact):
                    factIndex = top_fact[item][j].item()
                    entityIndex = self.fact2entityClass.fact2entity(factIndex)
                    name = str(entityIndex) + "-" + str(top_rel[item][i].item())
                    if name in fact_relation_to_fact.keys():
                        for k in fact_relation_to_fact[name]:
                            adj[item][self.args.FVQA.max_ans+j][k] = 1
                            graph[item][self.args.FVQA.max_ans+j][k] = top_rel[item][i].item() + 1

        for item in range(predicts_rel.shape[0]):
            for m in range(predicts_ans.shape[0]):
                adj[item][m][m] = 1
                graph[item][m][m] = predicts_ans[item][m].item()
        ##############################################################################

        graph = graph.cuda()
        adj = adj.cuda()
        return adj,graph

    def _get_input(self, predicts_rel, predicts_fact, predicts_ans,fact_relation_to_fact):
        #  ([128, 500])([128, 103])([128, 2791])

        # 把每一行的 predicts_rel 和 predicts_fact 中 top 拿出来凑对并且从fact_relation_to_ans筛选，将对应位置置0。rel top10，fact top30
        _, top_rel = predicts_rel.topk(self.args.top_rel, dim=1)
        _, top_fact = predicts_fact.topk(self.args.top_fact, dim=1)

        minmask = 1000
        minMask = torch.zeros(predicts_rel.shape[0])

        ## get input for MLP
        mask = torch.zeros(predicts_rel.shape[0], 500) #[128,500]
        
        predicts_ans = predicts_ans.float()
        # logger.info(f'#### type:={mask.type(),predicts_ans.type()}')
        for item in range(predicts_rel.shape[0]):
            for i in range(self.args.top_rel):
                for j in range(self.args.top_fact):
                    factIndex = top_fact[item][j].item()
                    entityIndex = self.fact2entityClass.fact2entity(factIndex)
                    name = str(top_fact[item][j].item()) + "-" + str(top_rel[item][i].item())
                    if name in fact_relation_to_fact.keys():
                        for k in fact_relation_to_fact[name]:
                            #mask[item][k] = 1
                            mask[item][k] = (predicts_rel[item][top_rel[item][i].item()].item()+predicts_fact[item][top_fact[item][j].item()])
                            if minMask[item] == 0:
                                minMask[item] = mask[item][k]
                            if minMask[item] > mask[item][k]:
                                minMask[item] = mask[item][k]
                            if minmask > mask[item][k]:
                                minmask = mask[item][k]
#                             if mask[item][k] < score:
#                                 mask[item][k] = score

        ##############################################################################
        #  predicts_ans[128,500]  mask[128,500]   adj[128,500]
#         mask = F.softmax(mask,dim =1)
#         temp = predicts_ans + mask
        for i in range(predicts_rel.shape[0]):
            for j in range(500):
                if mask[i][j] >= minMask[i]:
                    mask[i][j] -= minMask[i] 
                    mask[i][j] += 5
        
        if self.temp:
            self.temp = False
            print(mask[0],predicts_ans[0])
        mask = mask.cuda()
        predicts_ans = predicts_ans.cuda()
#         input = torch.cat((predicts_ans,mask),1)
#         input = input.cuda()
        return predicts_ans,mask

    def get_min_and_max(self, predicted):
        self.get_min_and_max_flag = 1
        ok, _ = predicted.topk(500, dim=1)
        min_tmp = ok[:, -1].reshape(1, -1)
        max_tmp = ok[:, 0].reshape(1, -1)
        # pdb.set_trace()
        if min(min_tmp[0]).item() < self.min:
            self.min = min(min_tmp[0]).item()
        if min(max_tmp[0]).item() > self.max:
            self.max = min(max_tmp[0]).item()

    def explain_experiments(self, predicts_rel, predicts_fact, predicts_ans, predicts_ans_orig, answers, fact_relation_to_ans, idx):

        if self.val_metrics.correct_1 >= self.val_metrics_orig.correct_1 and self.val_metrics.correct_1 > 0:
            # get the top 10 ans
            # get the top 10 relation/fact
            pre_h_r = []
            pre_t = []
            _, top_rel = predicts_rel.topk(self.args.top_rel, dim=1)
            _, top_fact = predicts_fact.topk(self.args.top_fact, dim=1)
            _, pre_ans = predicts_ans.data.topk(10, dim=1)
            _, pre_ans_orig = predicts_ans_orig.data.topk(10, dim=1)
            if len(answers.data.shape) == 3:
                answers = answers.data[0]
            _, real_ans = answers.topk(1, dim=1)

            top_rel = top_rel[0].cpu().numpy().tolist()
            top_fact = top_fact[0].cpu().numpy().tolist()
            pre_ans = pre_ans[0].cpu().numpy().tolist()
            pre_ans_orig = pre_ans_orig[0].cpu().numpy().tolist()
            idx = idx[0].item()
            real_ans = real_ans[0][0].item()

            for i in range(self.args.top_rel):
                for j in range(self.args.top_fact):
                    name = str(top_fact[j]) + "-" + str(top_rel[i])
                    if name in fact_relation_to_ans.keys():
                        for k in fact_relation_to_ans[name]:
                            if k in pre_ans:
                                name_real = self.fact_id[top_fact[j]] + "-" + self.relation_id[top_rel[i]]
                                pre_h_r.append(name_real)
                                pre_t.append(self.answer_id[k])

            top_rel_real = self.get_correspond_name(top_rel, self.relation_id)
            top_fact_real = self.get_correspond_name(top_fact, self.fact_id)
            pre_ans_real = self.get_correspond_name(pre_ans, self.answer_id)
            pre_ans_orig_real = self.get_correspond_name(pre_ans_orig, self.answer_id)
            real_ans = self.answer_id[real_ans]
            if real_ans != self.test_order_data[str(idx)]["answer"]:
                pdb.set_trace()
            question = self.test_order_data[str(idx)]["question"]
            question_id = idx
            real_fact_in_dataset = self.test_order_data[str(idx)]["fact"]
            image = self.test_order_data[str(idx)]["img_file"]
            pre_t = list(set(pre_t))
            pre_h_r = list(set(pre_h_r))
            # if real_fact_in_dataset[1] in top_rel_real and (real_fact_in_dataset[0] in top_fact_real or real_fact_in_dataset[1] in top_fact_real):
            if real_ans not in pre_t and len(pre_t) >= 1:
                logger.info('#################################################################################')
                logger.info(f'test id = {question_id}, question = {question}, img = {image}')
                logger.info(f'real suppord fact in dataset={real_fact_in_dataset}, real answer = {real_ans}')
                logger.info(f'normal model predict = {pre_ans_orig_real}')
                logger.info(f'our model predict = {pre_ans_real}')
                logger.info(f'our model predict relation = {top_rel_real}')
                logger.info(f'our model predict fact = {top_fact_real}')
                logger.info(f'suppord fact predict = {pre_h_r}')
                logger.info(f'correspond target = {pre_t}')
                logger.info('#################################################################################')
        # else:
        #     logger.info(f'fail')

        self.val_metrics.correct_1 = 0
        self.val_metrics_orig.correct_1 = 0
        self.val_metrics.correct_3 = 0
        self.val_metrics_orig.correct_3 = 0

    def get_entity_id(self):
        self.args.TEST.batch_size = 1
        exp_data = osp.join(self.args.data_root, "fvqa", "exp_data")
        num = "3"
        test_order_data = osp.join(exp_data, "test_data", "test" + num, "all_qs_dict_release_test_500_inorder.json")
        relation_id_path = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.relation.500.json")
        fact_id_path = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.fact.500.json")
        answer_id_path = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.500.json")
        with open(test_order_data, 'r') as fd:
            self.test_order_data = json.load(fd)
        with open(relation_id_path, 'r') as fd:
            self.relation_id = json.load(fd)
            self.relation_id = self.relation_id['answer']
            self.relation_id = self.invert_dict(self.relation_id)
        with open(fact_id_path, 'r') as fd:
            self.fact_id = json.load(fd)
            self.fact_id = self.fact_id['answer']
            self.fact_id = self.invert_dict(self.fact_id)
        with open(answer_id_path, 'r') as fd:
            self.answer_id = json.load(fd)
            self.answer_id = self.answer_id['answer']
            self.answer_id = self.invert_dict(self.answer_id)

    def get_correspond_name(self, id_list, dict):
        target = []
        for i in id_list:
            target.append(dict[i])
        return target

    def invert_dict(self, d):
        return {v: k for k, v in d.items()}

    def _model_choice(self, args):
        # models api
        fusion_model = getattr(fusion_net, args.fusion_model)(args, self.train_loader.dataset,
                                                              self.question_word2vec).cuda()
        assert args.answer_embedding in ['MLP']
        answer_model = getattr(answer_net, args.answer_embedding)(args, self.train_loader.dataset).cuda()
        return fusion_model, answer_model

    def _load_model(self, model, function, type_name):
        assert function in ["fusion", "embedding"]
        assert type_name in ["answer", "relation", "fact"]
        target = type_name
        model_name = type(model).__name__
        if not self.args.ZSL:
            target = "general_" + target
        save_path = os.path.join(self.args.FVQA.model_save_path, function)
        save_path = os.path.join(save_path, f'{target}_{model_name}_{self.args.FVQA.data_choice}.pkl')

        model.load_state_dict(torch.load(save_path,map_location='cuda:0'),)
        print(f"loading {save_path} model done!")

    def _save_model(self, model, function):
        model_name = type(model).__name__
        target = 'mlp'
        function = 'gcn'
        save_path = os.path.join(self.args.FVQA.model_save_path, function)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{target}_{model_name}_{self.args.FVQA.data_choice}.pkl')

        torch.save(model.state_dict(), save_path)
        return save_path

    def _get_fact_relation_dict(self):
        with open(self.args.FVQA.fact_relation_to_ans_path, 'r') as fd:
            fact_relation_to_ans = json.load(fd)
        return fact_relation_to_ans

    def _get_fact_relation_fact_dict(self):
        with open(self.args.FVQA.fact_relation_to_fact_path, 'r') as fd:
            fact_relation_to_fact = json.load(fd)
        return fact_relation_to_fact

    def _get_mask_on_ans(self, predicts_rel, predicts_fact, fact_relation_to_ans):
        # self.train_loader.dataset 必须是relation和fact，这样才能映射到relation和fact上。不然都是 128 * 500
        mask_ans = torch.ones(predicts_rel.shape[0], self.args.FVQA.max_ans)

        # 把每一行的 predicts_rel 和 predicts_fact 中 top 拿出来凑对并且从fact_relation_to_ans筛选，将对应位置置0。rel top10，fact top30
        _, top_rel = predicts_rel.topk(self.args.top_rel, dim=1)
        _, top_fact = predicts_fact.topk(self.args.top_fact, dim=1)

        # tmp = 0
        ##############################################################################
        # TODO: optimize this process via relation matric mapping rather than “ for-loop”
        for item in range(predicts_rel.shape[0]):
            for i in range(self.args.top_rel):
                for j in range(self.args.top_fact):
                    name = str(top_fact[item][j].item()) + "-" + str(top_rel[item][i].item())
                    if name in fact_relation_to_ans.keys():
                        for k in fact_relation_to_ans[name]:
                            # if mask_ans[item][k] > 0.1:
                            #     tmp += 1
                            # remove the mask from those ans ...
                            mask_ans[item][k] = 0

        # avg = tmp / self.args.TEST.batch_size
        # pdb.set_trace()
        # add mask
        mask_ans = mask_ans * -1 * self.args.soft_score
        mask_ans = mask_ans.cuda()
        ##############################################################################
        return mask_ans

    def _updata_best_result(self, max_acc, metrics):
        max_acc[3] = metrics.acc_all
        max_acc[2] = metrics.acc_10
        max_acc[1] = metrics.acc_3
        max_acc[0] = metrics.acc_1
if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    set_seed(cfg.random_seed)

    logger = initialize_exp(cfg)
    logger_path = get_dump_path(cfg)
    if not cfg.no_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    torch.cuda.set_device(cfg.gpu_id)
    runner = Runner(cfg)
    runner.run()

    #  information output:
    logger.info(f"best performance = {runner.max_acc[0]: .2f},{runner.max_acc[1]: .2f},{runner.max_acc[2]: .2f}. best epoch = {runner.best_epoch}, correspond_loss={runner.correspond_loss: .4f}")
    if not cfg.no_tensorboard:
        writer.close()