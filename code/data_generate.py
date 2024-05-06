import os,json,io
from pattern.en import parse,pprint


def check():
    with open('./data/KG_VQA/fvqa/exp_data/train_data/train3/all_qs_dict_release_train_500.json','r',encoding='utf8') as f:
        train_data = json.load(f)
    with open('./data/KG_VQA/fvqa/exp_data/common_data/answer.vocab.fvqa.fact.500.json','r',encoding='utf8') as f:
        fact_dict = json.load(f)


    allQ = []
    for key,value in train_data.items():
        fact = value['fact']
        question = value['question']
        if fact[0] in question:
            temp = question.replace(fact[0],'<none>')
            if temp not in allQ:
                allQ.append(temp)
        if fact[2] in question:
            temp = question.replace(fact[2],'<none>')
            if temp not in allQ:
                allQ.append(temp)
    for one in allQ:
        print(one)
    print(len(allQ))

    for key,value in fact_dict['answer'].items():
        print(key,value)

def parse():
    pass