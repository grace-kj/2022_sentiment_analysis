import os
import re
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import sklearn
import copy
import argparse
import numpy as np
import random
import torch
import time
import pickle
import gzip
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords  
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm,trange
from model import *
from masker import BertMasker,SentiBertMasker_tfidf
from pmi import Agument_PMI
nltk.download('stopwords')

MODEL_CLASS = {
    "roberta_reformulate"   : (RoBERTa_Reformulate, RobertaTokenizerFast),
    "roberta_mlm"           : (RoBERTa_MLM, RobertaTokenizerFast),
    "roberta"               : (RobertaForSequenceClassification, RobertaTokenizerFast),
    "bert_mlm"              : (BERT_MLM, BertTokenizerFast),
    "bert"                  : (BertForSequenceClassification, BertTokenizerFast),

}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        args.device = torch.device('cpu')
        print('Device name: cpu')
        
def load_data(args):
    data = {"train" : list(),"dev" : list(),"test" : list() }    
    labels = {"train" : list(),"dev" : list(),"test" : list() }    

    if not args.eda and not args.data_2000:
        with gzip.open("./data/" + args.data + "/data.train", mode="r") as inp:
            read_data = pickle.load(inp)   
            data["train"]   = [d for d, l in read_data]         
            labels["train"] = [l for d, l in read_data]         
        print("\n데이터 수 : {}".format(len(data["train"])))

    elif args.data_2000 and args.eda:
        with gzip.open("./data/" + args.data + "/data_full_with_eda.train".format(args.eda), mode="r") as inp:
            read_data = pickle.load(inp)   
            data["train"]   = [d for d, l in read_data]         
            labels["train"] = [l for d, l in read_data]
        print("\nEDA 5000: {}".format(len(data["train"])))

    elif args.data_2000:
        with gzip.open("./data/" + args.data + "/data_2000.train".format(args.eda), mode="r") as inp:
            read_data = pickle.load(inp)   
            data["train"]   = [d for d, l in read_data]         
            labels["train"] = [l for d, l in read_data]
        print("\nEDA 1000: {}".format(len(data["train"])))


    with gzip.open("./data/" + args.data + "/data.dev", mode="r") as inp:
        read_data = pickle.load(inp)   
        data["dev"]   = [d for d, l in read_data]         
        labels["dev"] = [l for d, l in read_data]         

    if args.data == 'SST-2_67k':
        data["test"] = data["dev"]
        labels["test"] = labels["dev"]
    else:
        with gzip.open("./data/" + args.data + "/data.test", mode="r") as inp:
            read_data = pickle.load(inp)   
            data["test"]   = [d for d, l in read_data]         
            labels["test"] = [l for d, l in read_data]         

    return data, labels

def save_checkpoint(args, best_epoch, best_model, optimizer, scheduler, acc_perplexity):
    if args.mode == 'post': best_model = best_model.roberta
    today = str(datetime.today().strftime("%Y%m%d"))
    output_dir = "./save_model/{}/{}_data{}_seed{}_lr{}_bs{}_ac{}_wd{}_len{}_epoch{}".format(args.mode, today, args.data,args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.weight_decay, args.max_seq_length, args.epochs)
    if (args.mode == 'post' and not args.swp) or args.mlm:   output_dir += "_MLM/"
    elif args.mode == 'post' and args.swp:     
        if args.add_name is not None:   output_dir += "_SWP_{}/".format(args.add_name)
        else:                           output_dir += "_SWP/"

    # elif args.mode == 'fine':                                   output_dir += "_fine_tunned{}/".format(round(acc_perplexity, 4))

    args.model_name_or_path = output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)                  
    model = best_model.module if hasattr(best_model, "module") else best_model    

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))


def load_checkpoint(checkpoint=None, model=None, optimizer=None, scheduler=None):
    print("=> Loading checkpoint")
    print("선언된 모델 : {}개".format(len(model.state_dict().keys())))
    print("저장된 모델 : {}개".format(len(checkpoint['state_dict'].keys())))

    matched_parameters = len([q for q in list(checkpoint['state_dict'].keys()) if q in list(model.state_dict().keys())])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("{}개 파라미터를 로드했습니다.".format(matched_parameters))
    if not optimizer is None:   optimizer.load_state_dict(checkpoint['optimizer'])
    if not scheduler is None:   scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler


def normal_tokenize(args, data, tokenizer,train=False):
    tokenized_data = list()
    pmi = list()
    length = list()
    tokens_num = list()    
    max_length = 0
    tokens_list = list()

    cls_token  = tokenizer.cls_token if not 'robert' in args.model else tokenizer.bos_token
    sep_token  = tokenizer.sep_token
    pad_token  = tokenizer.pad_token
    mask_token = tokenizer.mask_token
    for text in tqdm(data, desc="Tokenize : ",leave=True, position=0):

        tokens = tokenizer.tokenize(text)

        pmi.append(tokens)
        tokens_num.extend(tokens)       
        length.append(len(tokens))
        input_ids = list()
        input_mask = list()
        if max_length < len(tokens):                max_length = len(tokens)

        # (7 tokens) <s> ... </s></s> It was [MASK] </s>
        if args.reformulate_entailment:
            if len(tokens) >= args.max_seq_length-6:    
                tokens = tokens[:args.max_seq_length-7]
            base_tokens = [cls_token] + tokens + [sep_token]
            tokens = base_tokens + [sep_token, "It","was", mask_token, sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)        
            input_mask = [1] * len(base_tokens) + [0] * (len(tokens) - len(base_tokens))

        # (2 tokens) <s> ... </s>
        else:
            if len(tokens) >= args.max_seq_length-1:    
                tokens = tokens[:args.max_seq_length-2]
            args.avg_length = (args.avg_length + len(tokens))/2
            tokens = [cls_token] + tokens + [sep_token]        
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = list(map(str, token_ids))
            tokens_list.append(token_ids)                            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)        
            input_mask = [1] * len(input_ids)

        padding_length = args.max_seq_length - len(input_ids)

        input_ids += [tokenizer.convert_tokens_to_ids(pad_token)] * padding_length           
        input_mask += [0] * padding_length
        
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        
        tokenized_data.append({'input_ids' : input_ids.copy(), 'input_mask': input_mask.copy()})

    if train:
        args.total_tokens = [len(p) for p in pmi ]
        args.total_tokens = sum(args.total_tokens)
        # args.total_tokens = len(tokens_num)
        print("\n전체 Token 갯수 : {}".format(args.total_tokens))
        print("(수정)전체 Token 갯수 : {}".format(len(tokens_num)))             
        print("(수정)중복 제거 Token 갯수 : {}".format(len(set(tokens_num))))
    
    
    print("\n평균 길이 : ", round(sum(length)/len(length),2))
    print("최대 길이 : ", max_length)
    return tokenized_data.copy(), pmi.copy(), tokens_list.copy()

def tokenize(args=None, data=None, tokenizer=None, labels=None, train=False, dev=False):
    tokenized_data, pmi, tokens_list = normal_tokenize(args, data, tokenizer, train)

    return tokenized_data.copy(), pmi.copy(), tokens_list.copy()

def augment_tfidf(tokenizer,original_words):
    tfidf = dict()
    for token in original_words.keys():        
        if token[0] == 'Ġ':
            case_1 = token
            case_3 = token[0] + token[1:]     
        else:
            case_1 = token
            case_3 = 'Ġ' + token
        cases = list(set([case_1, case_3]))
        for case in cases:
            ids = tokenizer.convert_tokens_to_ids(case)
            if ids == 3:    # <unk> token
                continue
            tfidf[case] = original_words[token]    # 임의로 0값 삽입
    return tfidf

def sentiwords_without_pmi(tokenizer,original_words):
    senti_words = dict()
    for token in original_words.keys():        
        if token[0] == 'Ġ':
            case_1 = token[1].lower() + token[2:]
            # case_2 = token[1].upper() + token[2:]                    
            case_3 = token[0] + token[1].lower() + token[2:]                    
            # case_4 = token[0] + token[1].upper() + token[2:]
        else:
            case_1 = token[0].lower() + token[1:]
            # case_2 = token[0].upper() + token[1:]
            case_3 = 'Ġ' + token[0].lower() + token[1:]
            # case_4 = 'Ġ' + token[0].upper() + token[1:]        
        # cases = list(set([case_1, case_2, case_3, case_4]))
        cases = list(set([case_1, case_3]))
        for case in cases:
            ids = tokenizer.convert_tokens_to_ids(case)
            if ids == 3:    # <unk> token
                continue

            senti_words[ids] = original_words[token]    # 임의로 0값 삽입
    return senti_words

def impotrant_words_agumentation_conver_ids(tokenizer,original_words):

    new_important_words = dict()
    for token, score in original_words.items():

        if len(token) == 1:
            case_1 = token.lower()
            case_2 = token.upper()                              
            cases = list(set([case_1, case_2]))            
        else:
            if token[0] == 'Ġ':
                if len(token[1:]) > 1:            
                    case_1 = token[1].lower() + token[2:]
                    case_2 = token[1].upper() + token[2:]                    
                    case_3 = token[0] + token[1].lower() + token[2:]                    
                    case_4 = token[0] + token[1].upper() + token[2:]
                else:
                    case_1 = token[1].lower()
                    case_2 = token[1].upper()                  
                    case_3 = token[0] + token[1].lower()              
                    case_4 = token[0] + token[1].upper()
            else:
                case_1 = token[0].lower() + token[1:]
                case_2 = token[0].upper() + token[1:]
                case_3 = 'Ġ' + token[0].lower() + token[1:]
                case_4 = 'Ġ' + token[0].upper() + token[1:]

            cases = list(set([case_1, case_2, case_3, case_4]))

        for case in cases:
            ids = tokenizer.convert_tokens_to_ids(case)
            if ids == 3:    # <unk> token
                continue

            if not ids in new_important_words.keys():
                new_important_words[ids] = score
            else: # 만약에 이미 저장된 값이라면
                tmp_score = new_important_words[ids]
                new_important_words[ids] = (tmp_score + score)/2
    return new_important_words

def preprocess(args=None, data=None, labels=None):
    num_labels = len(set(labels['train'] + labels['dev'] + labels['test']))
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = num_labels
    print("\n##Config Num Labels : {} {}".format(config.num_labels, set(labels['train'] + labels['dev'] + labels['test'])))
    
    model_name = args.model.split('-')[0].lower()
    if args.reformulate_entailment:
        model_name = "{}_reformulate".format(model_name)
    else:
        model_name = "{}_mlm".format(model_name) if args.mlm else model_name

    model_class, tokenizer_class = MODEL_CLASS[model_name]
    config.is_decoder = False 
    if args.mode == 'fine':
        if args.model_name_or_path is not None:     
            saved_model = args.model_name_or_path + "/best_model.bin"
            print("\n학습된 모델 있음! : {}".format(saved_model))
            model = model_class.from_pretrained(pretrained_model_name_or_path=saved_model, config=config)            

        else:                                       
            print("\n학습된 모델 없음!")
            print("Require Model : {}\n".format(model_name))       
            model = model_class.from_pretrained(pretrained_model_name_or_path=args.model, config=config)


    elif args.mode == 'post':
        args.cfg = dict()               
        if args.model_name_or_path is not None:     
            saved_model = args.model_name_or_path + "/best_model.bin"            
            args.cfg['model'] = saved_model
            print("\n학습된 모델 있음! : {}".format(saved_model))

        else:                                       args.cfg['model'] = 'roberta-large'
        args.cfg['config'] = config
        model = ROBERTA_POST(args.cfg)

    # 3-2 Define Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model)
    if args.dummy_emb_v2:
        # https://github.com/huggingface/tokenizers/issues/247
        print("New Token 추가 전 : {}".format(model.resize_token_embeddings(len(tokenizer))))
        special_tokens_dict = {'additional_special_tokens': ['<pos>','<neg>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("New Token 추가 후 : {}".format(model.resize_token_embeddings(len(tokenizer))))

    # Run Tokenizerf
    tokenized_data = dict()
    # data['train'] = data['train'][:40]
    # labels['train'] = labels['train'][:40]    

    # data['dev'] = data['dev'][:10]
    # labels['dev'] = labels['dev'][:10]    

    # data['test'] = data['test'][:100]
    # labels['test'] = labels['test'][:100]    
    senti_words = dict()
    important_tokens = dict()
    tokens_list = list()
    tokenized_data['train'], pmi, tokens_list = tokenize(args, data=data['train'], tokenizer=tokenizer, labels=labels['train'], train=True)
    tokenized_data['dev'],  _, _ =  tokenize(args, data=data['dev'],  tokenizer=tokenizer, labels=labels['dev'], dev = True)
    tokenized_data['test'], _, _ =  tokenize(args, data=data['test'], tokenizer=tokenizer, labels=labels['test'])
    
#     if not (os.path.isfile("./data/{}/tokenized_data.train".format(args.data)) and 
#         os.path.isfile("./data/{}/tokenized_data.dev".format(args.data)) and 
#         os.path.isfile("./data/{}/tokenized_data.test".format(args.data)) and
#         os.path.isfile("./data/{}/tokenized_data.info".format(args.data))
#         ):
        
#         tokenized_data['train'], pmi, tokens_list = tokenize(args, data=data['train'], tokenizer=tokenizer, labels=labels['train'], train=True)



#         tokenized_data['dev'],  _, _ =  tokenize(args, data=data['dev'],  tokenizer=tokenizer, labels=labels['dev'], dev = True)
#         tokenized_data['test'], _, _ =  tokenize(args, data=data['test'], tokenizer=tokenizer, labels=labels['test'])
#         with gzip.open("./data/{}/tokenized_data.train".format(args.data), mode="w") as out:        pickle.dump(tokenized_data['train'], out)
#         with gzip.open("./data/{}/tokenized_data.dev".format(args.data), mode="w") as out:          pickle.dump(tokenized_data['dev'], out)
#         with gzip.open("./data/{}/tokenized_data.test".format(args.data), mode="w") as out:         pickle.dump(tokenized_data['test'], out)
#         with gzip.open("./data/{}/tokenized_data.info".format(args.data), mode="w") as out:         pickle.dump(args.total_tokens, out)

#     else:
#         print("\n [ Find pre-tokenized data ]")
#         with gzip.open("./data/{}/tokenized_data.train".format(args.data), mode="r") as inp:        tokenized_data['train'] = pickle.load(inp)
#         with gzip.open("./data/{}/tokenized_data.dev".format(args.data), mode="r") as inp:          tokenized_data['dev'] = pickle.load(inp)
#         with gzip.open("./data/{}/tokenized_data.test".format(args.data), mode="r") as inp:         tokenized_data['test'] = pickle.load(inp)
#         with gzip.open("./data/{}/tokenized_data.info".format(args.data), mode="r") as inp:         args.total_tokens = pickle.load(inp)

    # 감정 단어 쪽에 있는 중요 단어들 삭제
    if args.tfidf:  
        with gzip.open("./data/scores/(SST)TF_IDF_2021_12_27.score", mode='r') as inp:
            args.tfidf = pickle.load(inp)
        # args.tfidf = augment_tfidf(tokenizer,args.tfidf)

    print("\nImportance Tokens Load!")    
    important_tokens = dict()
    if args.weighted_masking_v3 or args.dummy_emb_v1:
        path = args.weights_name_or_path
        print("\n현재 읽은 중요도 파일 : {}".format(path))
        with gzip.open("./data/scores/{}".format(path), mode="r") as inp:
            important_tokens = pickle.load(inp)
            print("(확장 전) important_tokens : {}\t/\t{}개\n".format(type(important_tokens), len(important_tokens)))        
            important_tokens = impotrant_words_agumentation_conver_ids(tokenizer, important_tokens)                        
            print("(확장 후) important_tokens : {}\t/\t{}개\n".format(type(important_tokens), len(important_tokens)))
    masker = {"train":dict(), "eval":dict()}
    masker['train']['sentiment_masking'] = SentiBertMasker_tfidf(args,tokenizer,senti_words, important_tokens)   
    masker['eval']['sentiment_masking'] = SentiBertMasker_tfidf(args,tokenizer,senti_words, important_tokens)
    masker['train']['random_masking'] = BertMasker(tokenizer)   
    masker['eval']['random_masking'] = BertMasker(tokenizer)

    # for idx, (key, value) in enumerate(sorted(important_tokens.items(), key=(lambda x: x[1]), reverse = False)):
    #     if idx > 100:
    #         print("Key : {} ({})".format(tokenizer.convert_ids_to_tokens(key), value))
    # import sys
    # sys.exit()
    return model, tokenized_data, masker, tokens_list.copy()

def idx_to_tensor(args, tokenized_data, labels, batch_size=None):

    # print("tokenized_data : {}".format(len(tokenized_data["train"])))
    # print("labels : {}".format(len(labels['train'])))


    train_inputs_ids_tensor = torch.tensor([data['input_ids'] for data in tokenized_data["train"]], dtype=torch.long)
    train_inputs_mask_tensor = torch.tensor([data['input_mask'] for data in tokenized_data["train"]], dtype=torch.long)

    dev_inputs_ids_tensor = torch.tensor([data['input_ids'] for data in tokenized_data["dev"]], dtype=torch.long)
    dev_inputs_mask_tensor = torch.tensor([data['input_mask'] for data in tokenized_data["dev"]], dtype=torch.long)

    test_inputs_ids_tensor = torch.tensor([data['input_ids'] for data in tokenized_data["test"]], dtype=torch.long)
    test_inputs_mask_tensor = torch.tensor([data['input_mask'] for data in tokenized_data["test"]], dtype=torch.long)



    train_labels_tensor = torch.tensor(labels["train"], dtype=torch.long)
    dev_labels_tensor   = torch.tensor(labels["dev"], dtype=torch.long)
    test_labels_tensor  = torch.tensor(labels["test"], dtype=torch.long)

    train_data  = TensorDataset(train_inputs_ids_tensor, train_inputs_mask_tensor, train_labels_tensor)
    dev_data    = TensorDataset(dev_inputs_ids_tensor,dev_inputs_mask_tensor, dev_labels_tensor)
    test_data   = TensorDataset(test_inputs_ids_tensor, test_inputs_mask_tensor,test_labels_tensor)


    if args.compute_score:
        print("Compute_Score -> Sequential Train Data Loader")
        train_dataloader    = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    train_dataloader    = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_dataloader      = DataLoader(dev_data, batch_size=20, shuffle=False)
    test_dataloader     = DataLoader(test_data, batch_size=20, shuffle=False)

    return train_dataloader,dev_dataloader, test_dataloader

masked_length = 0

def batch_cuda(args=None, batch=None, polarity_labels=None, masked_labels=None, masked_type_labels=None):
    try:
        batch = tuple(t.to(args.device) for t in batch)
    except Exception as e:
        print("Error : {}".format(e))

    if polarity_labels is not None: polarity_labels = polarity_labels.to(args.device)
    if masked_labels is not None:   masked_labels = masked_labels.to(args.device)
    if masked_type_labels is not None: masked_type_labels = masked_type_labels.to(args.device)

    if args.mode == 'fine' and args.mlm:    inputs = {"args" : args, "input_ids" : batch[0], "attention_mask" : batch[1], 'labels' : batch[2], "masked_lm_labels" : masked_labels}
    elif args.mode == 'fine':               inputs = {"input_ids" : batch[0], "attention_mask" : batch[1], 'labels' : batch[2]}


    return inputs

def train_set(args, batch, train_masker, epoch=0, masked_token_num=None, train=False):
    masked_labels = None
    if args.mode == 'fine' and args.mlm and train:
        if args.weighted_masking_v3 and not args.tfidf == False:    
            batch[0], masked_labels, masked_type_labels = train_masker['sentiment_masking'].mask_tokens(inputs=batch[0], mlm_probability=args.mlm_probability,epoch=epoch)                            
            inputs = batch_cuda(args=args, batch=batch, masked_labels=masked_labels, masked_type_labels=masked_type_labels)

        elif args.weighted_masking_v3:    
            batch[0], masked_labels, masked_type_labels = train_masker['sentiment_masking'].mask_tokens(batch[0], args.mlm_probability,epoch=epoch)                            
            inputs = batch_cuda(args=args, batch=batch, masked_labels=masked_labels, masked_type_labels=masked_type_labels)

        else:                               
            batch[0], masked_labels     = train_masker['random_masking'].mask_tokens(batch[0], args.mlm_probability)        
            inputs = batch_cuda(args=args, batch=batch, masked_labels=masked_labels)

    elif args.mode == 'fine':
        inputs = batch_cuda(args=args, batch=batch)

    elif args.mode == 'post':
        if args.swp:    batch[0], masked_labels, masked_type_labels = train_masker['sentiment_masking'].mask_tokens(batch[0], args.mlm_probability)                
        else:           batch[0], masked_labels = train_masker['random_masking'].mask_tokens(batch[0], args.mlm_probability)
        inputs = batch_cuda(args=args, batch=batch,masked_labels=masked_labels)

    if masked_token_num is not None and masked_labels is not None:
        count_labels = masked_labels.flatten().detach().numpy().tolist()
        # count_labels = count_labels.numpy()
        # count_labels = count_labels.tolist()
        masked_token_num['total_tokens'] += len([l for l in count_labels if not l == -100])        
        masked_token_num['sentiment_tokens'] += train_masker['sentiment_masking'].total_masked_senti
        masked_token_num['common_tokens'] += train_masker['sentiment_masking'].total_masked_common
        masked_token_num['important_tokens'] += train_masker['sentiment_masking'].total_masked_important
        train_masker['sentiment_masking'].total_masked_senti = 0
        train_masker['sentiment_masking'].total_masked_common = 0
        train_masker['sentiment_masking'].total_masked_important = 0

    return inputs, masked_token_num



def default_parser(parser):
    # fixed config
    parser.add_argument("--epochs", type=int, default=5, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="max_seqeunce_length")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning_rate")
    parser.add_argument("--seed", type=int, default=1, help="random_seed")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight_decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient_accumulation_step")
    parser.add_argument("--warmup_proportion", type=float, default=0.06, help="warmup_proportion")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6, help="adam_epsilon")
    parser.add_argument("--device", type=str, default=None, help="device")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="mlm_probability")
    parser.add_argument("--data", type=str, default="SST-2", help="dataset")
    parser.add_argument("--mode", type=str, default="fine", help="fine_tuning / post_training")
    parser.add_argument("--model", type=str, default="roberta-large", help="roberta-large")

    parser.add_argument("--swp", action='store_true', help='sentiment word prediction')
    parser.add_argument("--mlm", action='store_true', help='Masked Language Model')
    parser.add_argument("--pmi", type=float, default=0.2, help="pmi")
    parser.add_argument("--sep", action='store_true', help='sep')

    # for 감정 단어 마스킹
    parser.add_argument("--sentiment_first", action='store_true', help='only_senti')
    parser.add_argument("--common_first", action='store_true', help='common_first')
    parser.add_argument("--additional_learning", action='store_true', help='negation')
    parser.add_argument("--important_tokens", action='store_true', help='important_token')
    parser.add_argument("--reformulate_entailment", action='store_true', help='reformulate downstream task to entailment')

    parser.add_argument("--cfg", type=str, default=None, help="config")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="saved_model")
    parser.add_argument("--weights_name_or_path", type=str, default=None, help="saved_weights")

    parser.add_argument("--add_name", type=str, default=None, help="additional_name")

    parser.add_argument("--train", action='store_true', help='train')
    parser.add_argument("--eval", action='store_true', help='eval')

    parser.add_argument("--total_tokens", type=int, default=0, help="number of token")

    parser.add_argument("--compute_score", action='store_true', help='compute_score')
    parser.add_argument("--extract_important_tokens", action='store_true', help='compute_score')
    parser.add_argument("--weighted_masking_v1", action='store_true', help='weighted_masking_v1')
    parser.add_argument("--weighted_masking_v2", action='store_true', help='weighted_masking_v2')
    parser.add_argument("--weighted_masking_v3", action='store_true', help='weighted_masking_v3')
    parser.add_argument("--mtl", type=int, default=1, help="MTL mode")

    parser.add_argument("--dummy_emb_v1", action='store_true', help='dummy_emb_v1')
    parser.add_argument("--dummy_emb_v2", action='store_true', help='dummy_emb_v2')

    parser.add_argument("--laplace_smoothing", type=float, default=0, help="laplace_smoothing")
    parser.add_argument("--mlm_loss", type=float, default=0.2, help="mlm_loss")

    # BM25처럼 써보려고 하는 거 성능이 안 좋으면 날려도 됨
    parser.add_argument("--avg_length", type=float, default=0, help="avg_length")
    parser.add_argument("--b", type=float, default=0.5, help="(0 ~ 1)")


    # parser.add_argument("--eda",  type=int, default=0, help="EDA: Easy data augmentation techniques for boosting perfor%02mance on text classification")
    parser.add_argument("--eda", action='store_true', help='EDA: Easy data augmentation techniques for boosting perfor%02mance on text classification')
    parser.add_argument("--data_2000", action='store_true', help='Load 2000 data')

    parser.add_argument("--tfidf", action='store_true', help='token-level TF-IDF')
    parser.add_argument("--compute_tfidf", action='store_true', help='token-level TF-IDF')

    parser.add_argument("--length_normalization", action='store_true', help='length_normalization')

    return parser    