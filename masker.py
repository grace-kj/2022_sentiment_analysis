import os
import numpy as np
import torch
import torch.nn as nn
import random
import math
from tqdm import tqdm
from collections import defaultdict


class Masker(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute_masked_indices(self, inputs, model, mlm_probability):
        raise NotImplementedError
    
    def gen_inputs_labels(self, inputs, masked_indices):
        raise NotImplementedError
        
    def mask_tokens(self, inputs, mlm_probability = 0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices = self.compute_masked_indices(inputs, mlm_probability)
        return self.gen_inputs_labels(inputs, masked_indices)

        
class BertMasker(Masker):
    def compute_masked_indices(self, inputs, mlm_probability):
        # inputs : (batch, max_seq_length)
        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.1500, 0.1500, ..., 0.1500]
        probability_matrix = torch.full(inputs.shape, mlm_probability)        

        # special_tokens_mask : batch (length)
        # special_tokens_mask[0] : max_seq_length (length)
        # special_tokens_mask[0] : [1, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
        # CLS, SEP, PAD 토큰은 Masking하지 않기 위해 1로 표시하는 듯 하다        
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]

        # pos_token = 50265
        # neg_token = 50266
        # for idx, tokens in enumerate(inputs.detach().numpy().tolist()):
        #     if pos_token in tokens or neg_token in tokens:
        #         print("\ninputs : {}".format(inputs[idx]))
        #         print("special_tokens_mask : {}".format(special_tokens_mask[idx]))

        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.0000, 0.1500, ..., 0.0000]
        # Special Token에 해당하는 위치는 확률을 0으로
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)    # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)

        # masked_indices : (batch, max_seq_length)
        # masked_indices[0] : tensor([False, True, False, False, True, ..., False, False])
        # 15% 확률로 bool type 변경 -> Masking할 단어 선정
        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)
        return masked_indices

    def gen_inputs_labels(self, inputs, masked_indices):
        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        # labels : (batch, max_seq_length)        
        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        inputs = inputs.clone()
        labels = inputs.clone()

        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SentiMasker(object):
    def __init__(self, args, tokenizer,senti_words, important_tokens=None):
        self.args = args
        self.tokenizer = tokenizer
        self.senti_words = senti_words
        self.important_tokens= important_tokens

        self.total_masked_senti = 0
        self.total_masked_common = 0  
        self.total_masked_important = 0          
        self.complete_case = 0
        self.fail_case = 0
        self.softmax = nn.Softmax(dim=0)

    def compute_masked_indices(self, inputs, mlm_probability):
        raise NotImplementedError
    
    def gen_inputs_labels(self, inputs, masked_indices):
        raise NotImplementedError
        
    def mask_tokens(self, inputs, mlm_probability = 0.15,epoch=None, inputs_num=None):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices, masked_type_labels = self.compute_masked_indices(inputs, mlm_probability,epoch, inputs_num)
        return self.gen_inputs_labels(inputs, masked_indices, masked_type_labels)




    # def get_polarity_labels(self, inputs, indices_senti_word,labels):
    #     # Q. Masking된 일반 Token도 Word Polarity Prediction을 해야 하는가?

    #     # 1. 감성단어만 WPP를 수행하겠다
    #     # 1-1. 일반 Token은 -100으로 Labels 설정
    #     polarity_labels = torch.full(inputs.shape, -100, dtype=torch.long)

    #     # 1-2. 감정 단어는 저장된 WP로 레이블 갱신
    #     boolean_senti_word = torch.tensor(indices_senti_word, dtype=torch.bool)
    #     for batch_num in range(len(boolean_senti_word)):
    #         for idx in range(len(boolean_senti_word[batch_num])):
    #             if boolean_senti_word[batch_num][idx] == True:
    #                 polarity_labels[batch_num][idx] = self.senti_words[inputs[batch_num][idx].item()][1]

    #     # 2. 마스킹된 일반 단어에 NEU 레이블링
    #     for batch_num in range(len(polarity_labels)):
    #         for idx in range(len(polarity_labels[batch_num])):
    #             # Masked Label인데 감성 단어로는 레이블링이 안되어 있는 경우 -> 마스킹된 일반 토큰
    #             if not labels[batch_num][idx] == -100 and polarity_labels[batch_num][idx] == -100:
    #                 # NEG : 0
    #                 # POS : 1
    #                 # NEU : 2
    #                 polarity_labels[batch_num][idx] = 2    
    #     print("polarity_labels : {}".format(polarity_labels))
    #     return polarity_labels  

class SentiBertMasker(SentiMasker):
    def compute_masked_tokens_num(self,inputs, mlm_probability):
        probability_matrix = torch.full(inputs.shape, mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) for ids in inputs.tolist()]                
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)    # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)                        

        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)
        masked_indices = torch.nonzero(masked_indices == True).detach().numpy().tolist()        
        # print("\n## probability_matrix : {}".format(probability_matrix))        
        # print("## masked_indices : {}".format(masked_indices))
        max_mask_per_batch = dict()
        for (batch_num, position) in masked_indices:
            if not batch_num in max_mask_per_batch:
                max_mask_per_batch[int(batch_num)] = 0
            max_mask_per_batch[int(batch_num)] += 1
        del probability_matrix
        return max_mask_per_batch

    def compute_masked_important_tokens_num(self,inputs, candidate_for_mask):        
        # 각 토큰 별 중요도를 이용해, 토큰을 마스킹
        # 입력으로 하나의 입력이 들어온다 (Batch 고려 X)
        masked_important_indices = list()
        unmasked_important_indices = list()

        # 01. 1 * Max Sequence length Tensor 생성
        probability_matrix = np.zeros((1, self.args.max_seq_length), dtype=bool)
        # print("probability_matrix : {} / {}".format(probability_matrix.shape, probability_matrix))

        # 02. special tokens 확률 0 (어차피 0으로 초기화되어있지만, 그냥 하는 것)
        # input은 하나의 Batch input

        pos_candidates = list()
        for token_ids in candidate_for_mask.keys():
            # 해당 Token이 문장 내 여러 Position에 위치한 경우
            try:
                for pos in candidate_for_mask[token_ids]:
                    # 점수를 넣을 때, 중요도가 높은 단어의 확률을 줄여야 한다. ex) Softmax
                    # probability_matrix[0][pos] = abs(1 - self.important_tokens[token_ids])
                    cur_important_score = abs(1-self.important_tokens[token_ids])
                    
                    probability_matrix[0][pos] = np.random.choice(a=[False, True], p=[1-cur_important_score, cur_important_score])

                    pos_candidates.append(pos)                    
            except Exception as e:
                print("\n########## [ ERROR ] ##########")
                print("E : {}".format(e))
                print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))

        
        masked_important_indices = torch.nonzero(torch.tensor(probability_matrix) == True).detach().numpy().tolist()        
        masked_important_indices = [pos for (batch_num, pos) in masked_important_indices]
        # # print("02. masked_important_indices : {}".format(masked_important_indices))

        unmasked_important_indices = [pos for pos in pos_candidates if not pos in masked_important_indices]
        # # 만에 하나, 15% 마스킹 갯수는 3개로 결정되었는데, 점수 기반으로 추출한 단어 후보 갯수는 그 미만일 수 있다.
        # # 즉 선정된 토큰 : [1,2,3] / 뽑히지 않은 토큰 : [4,5,6] 이렇게 두 개를 리턴해야한다.
        del probability_matrix        
        return masked_important_indices, unmasked_important_indices

    def compute_masked_important_tokens_num_v3(self, candidate_for_mask=None, need_to_mask_num=None, inputs_num=None):                
        # 각 토큰 별 중요도를 이용해, 토큰을 마스킹
        # 입력으로 하나의 입력이 들어온다 (Batch 고려 X)
        masked_important_indices = list()
        unmasked_important_indices = list()

        # 01. 1 * Max Sequence length Tensor 생성
        probability_matrix = torch.zeros((1, self.args.max_seq_length), dtype=torch.float32)

        # print("probability_matrix : {} / {}".format(probability_matrix.shape, probability_matrix))

        # 02. special tokens 확률 0 (어차피 0으로 초기화되어있지만, 그냥 하는 것)
        # input은 하나의 Batch input

        pos_candidates = list()
        end_pos = 0
        for token_ids in candidate_for_mask.keys():
            # 해당 Token이 문장 내 여러 Position에 위치한 경우
            try:
                for pos in candidate_for_mask[token_ids]:
                    # print("\nPOS : {}".format(pos))
                    # 점수를 넣을 때, 중요도가 높은 단어의 확률을 줄여야 한다. ex) Softmax
                    probability_matrix[0][pos] = abs(1-self.important_tokens[token_ids])                    
                    # print("probability_matrix[0][pos] : {}".format(probability_matrix[0][pos]))

                    # 코드를 아래와 같이 넣는 경우, 중요한 단어가 더 많이 마스킹된다.
                     # probability_matrix[0][pos] = abs(self.important_tokens[token_ids])                    
                    
                    pos_candidates.append(pos)
                    if pos > end_pos: end_pos = pos                                  
            except Exception as e:
                print("\n########## [ ERROR ] ##########")
                print("E : {}".format(e))
                print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))

        # print("01 probability_matrix : {}".format(probability_matrix))
        # print("02 probability_matrix : {}".format(probability_matrix))

        if not self.args.laplace_smoothing == 0:
            probability_matrix = probability_matrix[0][1:end_pos+1]
            down = torch.sum(probability_matrix).item()
            probability_matrix = probability_matrix.detach().numpy().tolist()

            for pos, probability in enumerate(probability_matrix):

                probability += self.args.laplace_smoothing
                # print("\n\nBefore : {} / {}".format(probability_matrix[pos], probability))
                down += (len(probability_matrix) * self.args.laplace_smoothing)
                probability_matrix[pos] = probability/down
                # print("After : {}".format(probability_matrix[pos]))
            

        else:
            probability_matrix = self.softmax(probability_matrix[0][1:end_pos+1])
            probability_matrix = probability_matrix.detach().numpy().tolist()

        masked_tokens = list()
        while True:
            if len(masked_tokens) >= need_to_mask_num: break

            std_score = 0
            rand_num = random.uniform(0.0, 1.0)
            for pos, probability_range in enumerate(probability_matrix):
                # 01. 특정 토큰이 아직 안 뽑혔고,
                # 02. 랜덤 난수가 특정 토큰에 할당된 범위에 속하는 경우
                # 03. CLS 토큰을 제외하므로, pos는 1부터 시작
                if (not pos+1 in masked_tokens) and (rand_num <= (std_score + probability_range)):
                    masked_tokens.append(pos+1)
                    break

                # std_score + probability_range
                # ex 1) 0 + 0.1
                # ex 2) 0.1 + 0.1.... 
                std_score += probability_range

        return masked_tokens.copy()

    def compute_masked_indices(self, inputs, mlm_probability,epoch=None, inputs_num=None):
        new_inputs = inputs.clone()

        masked_sentiment_tokens = list()
        masked_common_tokens = list()
        masked_important_tokens = list()
        masked_type_labels = list()

        # RoBERTa
        cls_token = self.tokenizer.cls_token if not 'robert' in self.args.model else self.tokenizer.bos_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token

        if self.args.dummy_emb_v2:       special_tokens = [cls_token,sep_token,pad_token, '<pos>', '<neg>'] # ['[CLS]','[SEP]','[PAD]']            
        else:                            special_tokens = [cls_token,sep_token,pad_token] # ['[CLS]','[SEP]','[PAD]']
        special_tokens_mask = [self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]        

        # 각 배치마다 몇 개의 토큰이 마스킹 되어야하는지 계산
        max_mask_per_batch = self.compute_masked_tokens_num(new_inputs, mlm_probability)

        # 마스킹될 감정 단어 선정 (10%)
        # 마스킹될 일반 단어 선정 (5%)
        # 같은 단어가 나올 위험이 있으므로, position으로 후보를 선정

        for batch_num, input_batch in enumerate(new_inputs):
            candidate_for_mask = defaultdict(list)                             # 감정/일반 단어 상관없이 마스킹될 수 있는 단어 목록
            input_batch = input_batch.detach().numpy().tolist()                # GPU 연산하는 게 아니므로, list로 변환
            for position, item in enumerate(input_batch):                      # 각 배치의 ids 값 출력
                if not item in special_tokens_mask:                            # 특수 토큰(<s>, </s>, <pad>)이 아닌 경우
                    candidate_for_mask[item].append(position) # {ids : position}

            if self.args.important_tokens and not self.args.weighted_masking_v1:
                sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]
                important_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.important_tokens.keys()]
                common_candidate_for_mask    = [position for ids, positions in candidate_for_mask.items() for position in positions if not ids in self.senti_words.keys() if not ids in self.important_tokens.keys()]                
            elif not self.args.weighted_masking_v1:
                sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]
                important_candidate_for_mask = []                
                common_candidate_for_mask    = [position for ids, positions in candidate_for_mask.items() for position in positions if not ids in self.senti_words.keys()]                

            # elif self.args.weighted_masking_v1 and self.args.swp:
            #     sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]

            masked_senti_token      = list()
            masked_common_token     = list()
            masked_important_token  = list()

            max_senti_mask      = 0
            max_common_mask     = 0
            max_important_mask  = 0

            success_fail_flag = True
            if batch_num in max_mask_per_batch: # 특정 Batch에는 마스킹된 단어가 없을 수 있다.
                if self.args.weighted_masking_v3:
                    # 깁스 샘플링 방식
                    masked_common_token = self.compute_masked_important_tokens_num_v3(candidate_for_mask.copy(), max_mask_per_batch[batch_num])

                elif self.args.weighted_masking_v2:
                    # 01. Masked될 후보 토큰 선정
                    # print("\n====\nmax_mask_per_batch[batch_num] : {}".format(max_mask_per_batch[batch_num]))    
                    masked_important_indices, unmasked_important_indices = self.compute_masked_important_tokens_num(new_inputs[batch_num], candidate_for_mask.copy())

                    # 02. 후보 토큰의 갯수가 Masking해야할 토큰보다 적은 경우
                    # 후보로 선택되지 않은 토큰들로 부족한 갯수를 채운다.
                    if max_mask_per_batch[batch_num] > len(masked_important_indices):
                        # print("\n갯수가 부족하다 {} / {}".format(max_mask_per_batch[batch_num], masked_important_indices))
                        need_tokens_num = max_mask_per_batch[batch_num] - len(masked_important_indices)
                        additional_extract_candidates = random.sample(unmasked_important_indices, need_tokens_num)                        
                        masked_important_indices.extend(additional_extract_candidates)

                    masked_important_indices = [position for ids, positions in candidate_for_mask.items() for position in positions]             
                    masked_common_token = random.sample(masked_important_indices, max_mask_per_batch[batch_num]) 
                    # 01. 토큰별 Impotrance Score 가져오기
                    # candidate_for_mask -> Special Token을 제외한 Tokens
                    # key 값은 ids

                elif self.args.weighted_masking_v1:

                    # 01. 토큰별 Impotrance Score 가져오기
                    # candidate_for_mask -> Special Token을 제외한 Tokens
                    # key 값은 ids
                    tokens_ids_importance = dict()
                    for token_ids in candidate_for_mask.keys():
                        try:
                            tokens_ids_importance[token_ids] = self.important_tokens[token_ids]
                        except:
                            print("\n########## [ ERROR ] ##########")
                            print("점수가 없는 token이 존재")
                            print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))
                            tokens_ids_importance[token_ids] = 100

                    # tokens_ids_importance = {token_ids : self.important_tokens[token_ids] for token_ids in candidate_for_mask.keys()}

                    # 02. 입력된 Token 가중치 오름차순 정렬 -> 덜 중요한 단어가 맨 앞으로
                    tokens_ids_importance = sorted(tokens_ids_importance.items(), key=(lambda x: x[1]), reverse = False)
                    tokens_ids_importance = [token_ids for token_ids, score in tokens_ids_importance]

                    # 03. 정렬된 Token 순서대로 마스킹 후보 선택                
                    for token_ids in tokens_ids_importance:
                        # candidate_for_mask -> {tokens ids : [pos1, pos2, ...]}
                        for position_list in candidate_for_mask[token_ids]:
                            if type(position_list) is list:
                                for pos in position_list:
                                    if self.args.swp and pos in sentiment_candidate_for_mask:
                                        continue
                                    masked_common_token.append(pos)
                                    # max_mask_per_batch[batch_num] -> 각 배치당 마스킹할 수 있는 갯수
                                    if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                                        break
                            else:
                                if self.args.swp and position_list in sentiment_candidate_for_mask:
                                    continue                                
                                masked_common_token.append(position_list)
                            # print("masked_common_token : {} / {}".format(len(masked_common_token), masked_common_token))
                            if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                                break
                        if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                            break


                elif self.args.important_tokens:
                    # quotient  = int(max_mask_per_batch[batch_num] / 3)
                    # remainder = int(max_mask_per_batch[batch_num] % 3)      
                    # max_senti_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    # max_important_mask = quotient * 1 + (1 if remainder > 1 else 0)                        

                    max_common_mask = int(max_mask_per_batch[batch_num])                      

                    while True:
                        # 01
                        # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        if (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(important_candidate_for_mask) >= max_important_mask
                            and len(common_candidate_for_mask) >= max_common_mask):
                                self.total_masked_important += max_important_mask                        
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                                
                                break
                        # 02
                        # (감정) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) < max_senti_mask
                            and len(important_candidate_for_mask) >= max_important_mask):
                            max_senti_mask -= 1
                            # 02 - 1
                            # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰 + 1
                            if len(important_candidate_for_mask) >= max_important_mask + 1:     max_important_mask += 1
                            else:                                                               max_common_mask += 1

                        # 03
                        # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(important_candidate_for_mask) < max_important_mask):
                            max_important_mask -= 1
                            # 03 - 1
                            # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰 + 1
                            if len(sentiment_candidate_for_mask) >= max_senti_mask + 1:     max_senti_mask += 1
                            else:                                                           max_common_mask += 1

                        # 04
                        # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) < max_senti_mask
                            and len(important_candidate_for_mask) < max_important_mask):
                            max_senti_mask -= 1                        
                            max_important_mask -= 1
                            max_common_mask += 2

                        # 05 일반토큰이 부족한 경우
                        elif len(common_candidate_for_mask) < max_common_mask:
                            max_common_mask -= 1                            
                            if len(important_candidate_for_mask) >= max_important_mask + 1:   max_important_mask += 1                            
                            elif len(sentiment_candidate_for_mask) >= max_senti_mask + 1:             max_senti_mask += 1


                        ## 부족하면 무조건 일반토큰으로 배정
                        # elif (len(sentiment_candidate_for_mask) < max_senti_mask
                        #     and len(important_candidate_for_mask) >= max_important_mask):
                        #     max_senti_mask -= 1
                        #     # 02 - 1
                        #     # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰 + 1
                        #     max_common_mask += 1

                        # # 03
                        # # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # # (중요) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        # elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                        #     and len(important_candidate_for_mask) < max_important_mask):
                        #     max_important_mask -= 1
                        #     # 03 - 1
                        #     # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰 + 1
                        #     max_common_mask += 1

                        # # 04
                        # # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # # (중요) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # elif (len(sentiment_candidate_for_mask) < max_senti_mask
                        #     and len(important_candidate_for_mask) < max_important_mask):
                        #     max_senti_mask -= 1                        
                        #     max_important_mask -= 1
                        #     max_common_mask += 2
                else:   # mix
                    # # 마스킹 확률이 15% 일 때는 나누기 3
                    quotient  = int(max_mask_per_batch[batch_num] / 3)
                    remainder = int(max_mask_per_batch[batch_num] % 3)

                    # # 아래는 감정단어 10% / 일반 단어 5%
                    max_senti_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    max_common_mask = quotient * 1 + (1 if remainder > 1 else 0)                        

                    # # 아래는 감정단어 5% / 일반 단어 10%
                    # quotient  = int(max_mask_per_batch[batch_num] / 3)
                    # remainder = int(max_mask_per_batch[batch_num] % 3)

                    # max_common_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    # max_senti_mask = quotient * 1 + (1 if remainder > 1 else 0)                        


                    # 아래는 감정단어 0% / 일반 단어 15%
                    # max_common_mask = int(max_mask_per_batch[batch_num])                      

                    # 감정단어 후보가 10%를 충족하지 못하거나, 그 반대의 경우가 발생할 수 있다.
                    while True:
                        if len(sentiment_candidate_for_mask) < max_senti_mask:
                            max_senti_mask -= 1
                            if len(common_candidate_for_mask) <= max_common_mask:
                                print("\n==================\nmax_mask_per_batch[batch_num] : {}".format(max_mask_per_batch[batch_num]))
                                print("\nquotient : {}".format(quotient))
                                print("remainder : {}".format(remainder))                                                

                                print("\n마스킹할 수 있는 감정 토큰 후보 : {}".format(len(sentiment_candidate_for_mask)))
                                print("마스킹할 수 있는 일반 토큰 후보 : {}".format(len(common_candidate_for_mask)))                
                                print("마스킹하려는 감정 토큰 갯수 : {}/{}".format(max_senti_mask, masked_senti_token))
                                print("마스킹하려는 일반 토큰 갯수 : {}/{}".format(max_common_mask,masked_common_token))                                
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                            
                                break
                            else:   max_common_mask += 1

                        elif len(common_candidate_for_mask) < max_common_mask:
                            max_common_mask -= 1                            

                            if len(sentiment_candidate_for_mask) <= max_senti_mask:
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                            
                                break
                            else:                                
                                max_senti_mask += 1


                        elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(common_candidate_for_mask) >= max_common_mask):
                            # 각 단어 유형별 마스킹 단어 추출     
                            self.total_masked_senti += max_senti_mask
                            self.total_masked_common += max_common_mask                            

                            if max_senti_mask > 0:          masked_type_labels.append(1)
                            elif max_senti_mask == 0:       masked_type_labels.append(2)
                            break
                
                if not self.args.weighted_masking_v1 and not self.args.weighted_masking_v2 and not self.args.weighted_masking_v3:
                    masked_senti_token  = random.sample(sentiment_candidate_for_mask, max_senti_mask)
                    masked_important_token = random.sample(important_candidate_for_mask, max_important_mask)                
                    masked_common_token = random.sample(common_candidate_for_mask, max_common_mask)

                # print("tokens num : {} / {}".format(len(masked_common_token) + len(masked_important_token) + len(masked_senti_token), max_mask_per_batch[batch_num]))

            else: # 마스킹된 토큰이 없는 Batch의 경우
                masked_type_labels.append(0)


            senti_indice        = [1 if position in masked_senti_token else 0 for position, token in enumerate(input_batch)]            
            important_indice    = [1 if position in masked_important_token else 0 for position, token in enumerate(input_batch)]            
            common_indice       = [1 if position in masked_common_token else 0 for position, token in enumerate(input_batch)]            

            # print("\nsenti_indice : {}/{}".format(type(senti_indice), senti_indice))
            # print("important_indice : {}/{}".format(type(important_indice), important_indice))
            # print("common_indice : {}/{}".format(type(common_indice), common_indice))      

            # print("senti_indice : {}".format(len([a for a in senti_indice if a == 1])))
            # print("common_indice : {}".format(len([a for a in common_indice if a == 1])))

            masked_sentiment_tokens.append(senti_indice.copy())
            masked_important_tokens.append(important_indice.copy())
            masked_common_tokens.append(common_indice.copy())

        # inputs : (batch, max_seq_length)
        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.1500, 0.1500, ..., 0.1500]        
        probability_matrix = torch.full(inputs.shape, 0)        

        # special_tokens_mask : batch (length)
        # special_tokens_mask[0] : max_seq_length (length)
        # special_tokens_mask[0] : [1, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
        # CLS, SEP, PAD 토큰은 Masking하지 않기 위해 1로 표시        
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]                

        
        # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)
        # print("masked_sentiment_tokens : {}/{}".format(type(masked_sentiment_tokens), masked_sentiment_tokens[0]))
        # print("masked_important_token : {}/{}".format(type(masked_important_token), masked_important_token[0]))
        # print("masked_common_tokens : {}/{}".format(type(masked_common_tokens), masked_common_tokens[0]))      

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)         # Special Token의 Masking 확률 0                        
        probability_matrix.masked_fill_(torch.tensor(masked_sentiment_tokens, dtype=torch.bool), value=1.0)     # Sentiment Token Masking       
        probability_matrix.masked_fill_(torch.tensor(masked_common_tokens, dtype=torch.bool), value=1.0)        # Common Token Masking
        if self.args.important_tokens:
            probability_matrix.masked_fill_(torch.tensor(masked_important_tokens, dtype=torch.bool), value=1.0)        # Common Token Masking        

        # masked_indices : (batch, max_seq_length)
        # 15% 확률을 바탕으로 1 or 0 예측 -> boolean type 변경        
        # masked_indices[0] : tensor([False, True, False, False, True, ..., False, False])
        # masked_indices 값 중, True는 Masking 대상
        probability_matrix = probability_matrix.float()
        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)

        return masked_indices, torch.tensor(masked_type_labels, dtype=torch.long)

    def gen_inputs_labels(self, inputs, masked_indices, masked_type_labels):
        original_inputs = inputs.clone()
        inputs = inputs.clone()
        labels = inputs.clone()

        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        # Masking이 안된 Token들은 -100
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # print("\n\nA : ", torch.bernoulli(torch.full(labels.shape, 0.8)))        
        # print("B : ", torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool))
        # print("C : ", torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices)
        # print("D : ", masked_indices)

        masked_candiate = torch.full(labels.shape, 0.8)
        
        # [MASK] : 103
        indices_replaced = torch.bernoulli(masked_candiate).type(torch.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # get_polarity_labels(self, inputs, indices_senti_word,labels):
        # polarity_labels = self.get_polarity_labels(original_inputs,indices_senti_word,labels)
        # return inputs, labels, polarity_labels           

        return inputs, labels, masked_type_labels


    def get_polarity_labels(self, inputs, marked_senti_pos,labels):
        # Q. Masking된 일반 Token도 Word Polarity Prediction을 해야 하는가?

        # 1. 감성단어만 WPP를 수행하겠다
        # 1-1. 일반 Token은 -100으로 Labels 설정
        polarity_labels = torch.full(inputs.shape, -100, dtype=torch.long)

        # 1-2. 감정 단어는 저장된 WP로 레이블 갱신
        boolean_senti_word = torch.tensor(marked_senti_pos, dtype=torch.bool)
        for batch_num in range(len(boolean_senti_word)):
            for idx in range(len(boolean_senti_word[batch_num])):
                if boolean_senti_word[batch_num][idx] == True:
                    polarity_labels[batch_num][idx] = self.senti_words[inputs[batch_num][idx].item()][1]

        # 2. 마스킹된 일반 단어에 NEU 레이블링
        for batch_num in range(len(polarity_labels)):
            for idx in range(len(polarity_labels[batch_num])):
                # Masked Label인데 감성 단어로는 레이블링이 안되어 있는 경우 -> 마스킹된 일반 토큰
                if not labels[batch_num][idx] == -100 and polarity_labels[batch_num][idx] == -100:
                    # NEG : 0
                    # POS : 1
                    # NEU : 2
                    polarity_labels[batch_num][idx] = 2    
        print("polarity_labels : {}".format(polarity_labels))
        return polarity_labels  



class SentiMasker_tfidf(object):
    def __init__(self, args, tokenizer,senti_words, important_tokens=None):
        self.args = args
        self.tokenizer = tokenizer
        self.senti_words = senti_words
        self.important_tokens= important_tokens

        self.total_masked_senti = 0
        self.total_masked_common = 0  
        self.total_masked_important = 0          
        self.complete_case = 0
        self.fail_case = 0
        self.softmax = nn.Softmax(dim=0)

    def compute_masked_indices(self, inputs, mlm_probability):
        raise NotImplementedError
    
    def gen_inputs_labels(self, inputs, masked_indices):
        raise NotImplementedError
        
    def mask_tokens(self, inputs, mlm_probability = 0.15,epoch=None, inputs_num=None):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        masked_indices, masked_type_labels = self.compute_masked_indices(inputs, mlm_probability,epoch, inputs_num)
        return self.gen_inputs_labels(inputs, masked_indices, masked_type_labels)


class SentiBertMasker_tfidf(SentiMasker_tfidf):
    def compute_masked_tokens_num(self,inputs, mlm_probability):
        probability_matrix = torch.full(inputs.shape, mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) for ids in inputs.tolist()]                
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)    # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)                        

        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)
        masked_indices = torch.nonzero(masked_indices == True).detach().numpy().tolist()        
        # print("\n## probability_matrix : {}".format(probability_matrix))        
        # print("## masked_indices : {}".format(masked_indices))
        max_mask_per_batch = dict()
        for (batch_num, position) in masked_indices:
            if not batch_num in max_mask_per_batch:
                max_mask_per_batch[int(batch_num)] = 0
            max_mask_per_batch[int(batch_num)] += 1
        del probability_matrix
        return max_mask_per_batch

    def compute_masked_important_tokens_num(self,inputs, candidate_for_mask):        
        # 각 토큰 별 중요도를 이용해, 토큰을 마스킹
        # 입력으로 하나의 입력이 들어온다 (Batch 고려 X)
        masked_important_indices = list()
        unmasked_important_indices = list()

        # 01. 1 * Max Sequence length Tensor 생성
        probability_matrix = np.zeros((1, self.args.max_seq_length), dtype=bool)
        # print("probability_matrix : {} / {}".format(probability_matrix.shape, probability_matrix))

        # 02. special tokens 확률 0 (어차피 0으로 초기화되어있지만, 그냥 하는 것)
        # input은 하나의 Batch input

        pos_candidates = list()
        for token_ids in candidate_for_mask.keys():
            # 해당 Token이 문장 내 여러 Position에 위치한 경우
            try:
                for pos in candidate_for_mask[token_ids]:
                    # 점수를 넣을 때, 중요도가 높은 단어의 확률을 줄여야 한다. ex) Softmax
                    # probability_matrix[0][pos] = abs(1 - self.important_tokens[token_ids])
                    cur_important_score = abs(1-self.important_tokens[token_ids])
                    
                    probability_matrix[0][pos] = np.random.choice(a=[False, True], p=[1-cur_important_score, cur_important_score])

                    pos_candidates.append(pos)                    
            except Exception as e:
                print("\n########## [ ERROR ] ##########")
                print("E : {}".format(e))
                print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))

        
        masked_important_indices = torch.nonzero(torch.tensor(probability_matrix) == True).detach().numpy().tolist()        
        masked_important_indices = [pos for (batch_num, pos) in masked_important_indices]
        # # print("02. masked_important_indices : {}".format(masked_important_indices))

        unmasked_important_indices = [pos for pos in pos_candidates if not pos in masked_important_indices]
        # # 만에 하나, 15% 마스킹 갯수는 3개로 결정되었는데, 점수 기반으로 추출한 단어 후보 갯수는 그 미만일 수 있다.
        # # 즉 선정된 토큰 : [1,2,3] / 뽑히지 않은 토큰 : [4,5,6] 이렇게 두 개를 리턴해야한다.
        del probability_matrix        
        return masked_important_indices, unmasked_important_indices

    def compute_masked_important_tokens_num_v3(self, candidate_for_mask=None, need_to_mask_num=None):        
        # 각 토큰 별 중요도를 이용해, 토큰을 마스킹
        # 입력으로 하나의 입력이 들어온다 (Batch 고려 X)
        masked_important_indices = list()
        unmasked_important_indices = list()

        # 01. 1 * Max Sequence length Tensor 생성
        probability_matrix = torch.zeros((1, self.args.max_seq_length), dtype=torch.float32)

        # print("probability_matrix : {} / {}".format(probability_matrix.shape, probability_matrix))

        # 02. special tokens 확률 0 (어차피 0으로 초기화되어있지만, 그냥 하는 것)
        # input은 하나의 Batch input

        pos_candidates = list()
        end_pos = 0

        cur_input_len = len(candidate_for_mask.keys())

        for token_ids in candidate_for_mask.keys():
            # 해당 Token이 문장 내 여러 Position에 위치한 경우
            # print("self.args.tfidf[token_ids]) : {}".format(self.args.tfidf[self.tokenizer.convert_ids_to_tokens(token_ids)]))
            try:
                for pos in candidate_for_mask[token_ids]:
                    # 점수를 넣을 때, 중요도가 높은 단어의 확률을 줄여야 한다. ex) Softmax
                    # probability_matrix[0][pos] = abs(1-self.important_tokens[token_ids])

                    # probability_matrix[0][pos] = (0.2 * abs(1 - tfidf[pos-1])) + (0.8 * abs(1-self.important_tokens[token_ids]))

                    # 기존 점수 + TFIDF (1:1) -> 성능 안 나옴
                    # probability_matrix[0][pos] = (abs(1-self.important_tokens[token_ids]) + (1-self.args.tfidf[token_ids]))/2
                    
                    probability_matrix[0][pos] = abs(1-self.important_tokens[token_ids]) * ((1-self.args.b) + self.args.b * (cur_input_len/self.args.avg_length))
                    # print("\n안뇽")
                    # print("Cur length : {}".format(cur_input_len))
                    # print("avg length : {}".format(self.args.avg_length))
                    # print("기존 점수 :{}".format(abs(1-self.important_tokens[token_ids])))                    
                    # print("새로운 점수 :{}".format(probability_matrix[0][pos]))                    

                    # 기존 점수                     
                    # only_tfidf
                    # probability_matrix[0][pos] = 1-self.args.tfidf[token_ids]

                    # print("probability_matrix[0][pos] : {}".format(probability_matrix[0][pos]))

                    # 코드를 아래와 같이 넣는 경우, 중요한 단어가 더 많이 마스킹된다.
                    # probability_matrix[0][pos] = abs(self.important_tokens[token_ids])                    
                    
                    pos_candidates.append(pos)
                    if pos > end_pos: end_pos = pos                                  
            except Exception as e:
                print("\n########## [ ERROR ] ##########")
                print("E : {}".format(e))
                print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))

        # print("01 probability_matrix : {}".format(probability_matrix))
        # print("02 probability_matrix : {}".format(probability_matrix))

        if not self.args.laplace_smoothing == 0:
            probability_matrix = probability_matrix[0][1:end_pos+1]
            down = torch.sum(probability_matrix).item()
            probability_matrix = probability_matrix.detach().numpy().tolist()

            for pos, probability in enumerate(probability_matrix):

                probability += self.args.laplace_smoothing
                # print("\n\nBefore : {} / {}".format(probability_matrix[pos], probability))
                down += (len(probability_matrix) * self.args.laplace_smoothing)
                probability_matrix[pos] = probability/down
                # print("After : {}".format(probability_matrix[pos]))
            

        else:
            probability_matrix = self.softmax(probability_matrix[0][1:end_pos+1])
            probability_matrix = probability_matrix.detach().numpy().tolist()

        masked_tokens = list()

        so_big_rand = 0
        while True:
            if len(masked_tokens) >= need_to_mask_num: break

            std_score = 0
            rand_num = random.uniform(0.0, 1.0)
            if so_big_rand >= 1:    rand_num = rand_num * (0.1 ** so_big_rand)
            for pos, probability_range in enumerate(probability_matrix):
                # 01. 특정 토큰이 아직 안 뽑혔고,
                # 02. 랜덤 난수가 특정 토큰에 할당된 범위에 속하는 경우
                # 03. CLS 토큰을 제외하므로, pos는 1부터 시작
                if (not pos+1 in masked_tokens) and (rand_num <= (std_score + probability_range)):
                    masked_tokens.append(pos+1)
                    so_big_rand = 0
                    pos = 0
                    break

                # std_score + probability_range
                # ex 1) 0 + 0.1
                # ex 2) 0.1 + 0.1.... 
                std_score += probability_range

            if (pos + 1) == len(probability_matrix):
                so_big_rand += 1
        return masked_tokens.copy()

    def compute_masked_indices(self, inputs, mlm_probability,epoch=None, inputs_num=None):

        new_inputs = inputs.clone()

        masked_sentiment_tokens = list()
        masked_common_tokens = list()
        masked_important_tokens = list()
        masked_type_labels = list()

        # RoBERTa
        cls_token = self.tokenizer.cls_token if not 'robert' in self.args.model else self.tokenizer.bos_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token

        if self.args.dummy_emb_v2:       special_tokens = [cls_token,sep_token,pad_token, '<pos>', '<neg>'] # ['[CLS]','[SEP]','[PAD]']            
        else:                            special_tokens = [cls_token,sep_token,pad_token] # ['[CLS]','[SEP]','[PAD]']
        special_tokens_mask = [self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]        

        # 각 배치마다 몇 개의 토큰이 마스킹 되어야하는지 계산
        max_mask_per_batch = self.compute_masked_tokens_num(new_inputs, mlm_probability)

        # 마스킹될 감정 단어 선정 (10%)
        # 마스킹될 일반 단어 선정 (5%)
        # 같은 단어가 나올 위험이 있으므로, position으로 후보를 선정

        for batch_num, input_batch in enumerate(new_inputs):
            candidate_for_mask = defaultdict(list)                             # 감정/일반 단어 상관없이 마스킹될 수 있는 단어 목록
            input_batch = input_batch.detach().numpy().tolist()                # GPU 연산하는 게 아니므로, list로 변환
            for position, item in enumerate(input_batch):                      # 각 배치의 ids 값 출력
                if not item in special_tokens_mask:                            # 특수 토큰(<s>, </s>, <pad>)이 아닌 경우
                    candidate_for_mask[item].append(position) # {ids : position}

            if self.args.important_tokens and not self.args.weighted_masking_v1:
                sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]
                important_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.important_tokens.keys()]
                common_candidate_for_mask    = [position for ids, positions in candidate_for_mask.items() for position in positions if not ids in self.senti_words.keys() if not ids in self.important_tokens.keys()]                
            elif not self.args.weighted_masking_v1:
                sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]
                important_candidate_for_mask = []                
                common_candidate_for_mask    = [position for ids, positions in candidate_for_mask.items() for position in positions if not ids in self.senti_words.keys()]                

            # elif self.args.weighted_masking_v1 and self.args.swp:
            #     sentiment_candidate_for_mask = [position for ids, positions in candidate_for_mask.items() for position in positions if ids in self.senti_words.keys()]

            masked_senti_token      = list()
            masked_common_token     = list()
            masked_important_token  = list()

            max_senti_mask      = 0
            max_common_mask     = 0
            max_important_mask  = 0

            success_fail_flag = True
            if batch_num in max_mask_per_batch: # 특정 Batch에는 마스킹된 단어가 없을 수 있다.
                if self.args.weighted_masking_v3:
                    # 깁스 샘플링 방식

                    masked_common_token = self.compute_masked_important_tokens_num_v3(candidate_for_mask.copy(), max_mask_per_batch[batch_num])
                elif self.args.weighted_masking_v2:
                    # 01. Masked될 후보 토큰 선정
                    # print("\n====\nmax_mask_per_batch[batch_num] : {}".format(max_mask_per_batch[batch_num]))    
                    masked_important_indices, unmasked_important_indices = self.compute_masked_important_tokens_num(new_inputs[batch_num], candidate_for_mask.copy())

                    # 02. 후보 토큰의 갯수가 Masking해야할 토큰보다 적은 경우
                    # 후보로 선택되지 않은 토큰들로 부족한 갯수를 채운다.
                    if max_mask_per_batch[batch_num] > len(masked_important_indices):
                        # print("\n갯수가 부족하다 {} / {}".format(max_mask_per_batch[batch_num], masked_important_indices))
                        need_tokens_num = max_mask_per_batch[batch_num] - len(masked_important_indices)
                        additional_extract_candidates = random.sample(unmasked_important_indices, need_tokens_num)                        
                        masked_important_indices.extend(additional_extract_candidates)

                    masked_important_indices = [position for ids, positions in candidate_for_mask.items() for position in positions]             
                    masked_common_token = random.sample(masked_important_indices, max_mask_per_batch[batch_num]) 
                    # 01. 토큰별 Impotrance Score 가져오기
                    # candidate_for_mask -> Special Token을 제외한 Tokens
                    # key 값은 ids

                elif self.args.weighted_masking_v1:

                    # 01. 토큰별 Impotrance Score 가져오기
                    # candidate_for_mask -> Special Token을 제외한 Tokens
                    # key 값은 ids
                    tokens_ids_importance = dict()
                    for token_ids in candidate_for_mask.keys():
                        try:
                            tokens_ids_importance[token_ids] = self.important_tokens[token_ids]
                        except:
                            print("\n########## [ ERROR ] ##########")
                            print("점수가 없는 token이 존재")
                            print("token_ids : {} / {}".format(token_ids, self.tokenizer.convert_ids_to_tokens(token_ids)))
                            tokens_ids_importance[token_ids] = 100

                    # tokens_ids_importance = {token_ids : self.important_tokens[token_ids] for token_ids in candidate_for_mask.keys()}

                    # 02. 입력된 Token 가중치 오름차순 정렬 -> 덜 중요한 단어가 맨 앞으로
                    tokens_ids_importance = sorted(tokens_ids_importance.items(), key=(lambda x: x[1]), reverse = False)
                    tokens_ids_importance = [token_ids for token_ids, score in tokens_ids_importance]

                    # 03. 정렬된 Token 순서대로 마스킹 후보 선택                
                    for token_ids in tokens_ids_importance:
                        # candidate_for_mask -> {tokens ids : [pos1, pos2, ...]}
                        for position_list in candidate_for_mask[token_ids]:
                            if type(position_list) is list:
                                for pos in position_list:
                                    if self.args.swp and pos in sentiment_candidate_for_mask:
                                        continue
                                    masked_common_token.append(pos)
                                    # max_mask_per_batch[batch_num] -> 각 배치당 마스킹할 수 있는 갯수
                                    if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                                        break
                            else:
                                if self.args.swp and position_list in sentiment_candidate_for_mask:
                                    continue                                
                                masked_common_token.append(position_list)
                            # print("masked_common_token : {} / {}".format(len(masked_common_token), masked_common_token))
                            if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                                break
                        if len(masked_common_token) >= max_mask_per_batch[batch_num]:
                            break


                elif self.args.important_tokens:
                    # quotient  = int(max_mask_per_batch[batch_num] / 3)
                    # remainder = int(max_mask_per_batch[batch_num] % 3)      
                    # max_senti_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    # max_important_mask = quotient * 1 + (1 if remainder > 1 else 0)                        

                    max_common_mask = int(max_mask_per_batch[batch_num])                      

                    while True:
                        # 01
                        # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        if (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(important_candidate_for_mask) >= max_important_mask
                            and len(common_candidate_for_mask) >= max_common_mask):
                                self.total_masked_important += max_important_mask                        
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                                
                                break
                        # 02
                        # (감정) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) < max_senti_mask
                            and len(important_candidate_for_mask) >= max_important_mask):
                            max_senti_mask -= 1
                            # 02 - 1
                            # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰 + 1
                            if len(important_candidate_for_mask) >= max_important_mask + 1:     max_important_mask += 1
                            else:                                                               max_common_mask += 1

                        # 03
                        # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(important_candidate_for_mask) < max_important_mask):
                            max_important_mask -= 1
                            # 03 - 1
                            # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰 + 1
                            if len(sentiment_candidate_for_mask) >= max_senti_mask + 1:     max_senti_mask += 1
                            else:                                                           max_common_mask += 1

                        # 04
                        # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # (중요) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        elif (len(sentiment_candidate_for_mask) < max_senti_mask
                            and len(important_candidate_for_mask) < max_important_mask):
                            max_senti_mask -= 1                        
                            max_important_mask -= 1
                            max_common_mask += 2

                        # 05 일반토큰이 부족한 경우
                        elif len(common_candidate_for_mask) < max_common_mask:
                            max_common_mask -= 1                            
                            if len(important_candidate_for_mask) >= max_important_mask + 1:   max_important_mask += 1                            
                            elif len(sentiment_candidate_for_mask) >= max_senti_mask + 1:             max_senti_mask += 1


                        ## 부족하면 무조건 일반토큰으로 배정
                        # elif (len(sentiment_candidate_for_mask) < max_senti_mask
                        #     and len(important_candidate_for_mask) >= max_important_mask):
                        #     max_senti_mask -= 1
                        #     # 02 - 1
                        #     # (중요) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰 + 1
                        #     max_common_mask += 1

                        # # 03
                        # # (감정) 마스킹할 수 있는 토큰  >= 마스킹해야하는 토큰
                        # # (중요) 마스킹할 수 있는 토큰  <  마스킹해야하는 토큰
                        # elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                        #     and len(important_candidate_for_mask) < max_important_mask):
                        #     max_important_mask -= 1
                        #     # 03 - 1
                        #     # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰 + 1
                        #     max_common_mask += 1

                        # # 04
                        # # (감정) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # # (중요) 마스킹할 수 있는 토큰  < 마스킹해야하는 토큰
                        # elif (len(sentiment_candidate_for_mask) < max_senti_mask
                        #     and len(important_candidate_for_mask) < max_important_mask):
                        #     max_senti_mask -= 1                        
                        #     max_important_mask -= 1
                        #     max_common_mask += 2
                else:   # mix
                    # # 마스킹 확률이 15% 일 때는 나누기 3
                    quotient  = int(max_mask_per_batch[batch_num] / 3)
                    remainder = int(max_mask_per_batch[batch_num] % 3)

                    # # 아래는 감정단어 10% / 일반 단어 5%
                    max_senti_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    max_common_mask = quotient * 1 + (1 if remainder > 1 else 0)                        

                    # # 아래는 감정단어 5% / 일반 단어 10%
                    # quotient  = int(max_mask_per_batch[batch_num] / 3)
                    # remainder = int(max_mask_per_batch[batch_num] % 3)

                    # max_common_mask  = quotient * 2 + (1 if remainder > 0 else 0)
                    # max_senti_mask = quotient * 1 + (1 if remainder > 1 else 0)                        


                    # 아래는 감정단어 0% / 일반 단어 15%
                    # max_common_mask = int(max_mask_per_batch[batch_num])                      

                    # 감정단어 후보가 10%를 충족하지 못하거나, 그 반대의 경우가 발생할 수 있다.
                    while True:
                        if len(sentiment_candidate_for_mask) < max_senti_mask:
                            max_senti_mask -= 1
                            if len(common_candidate_for_mask) <= max_common_mask:
                                print("\n==================\nmax_mask_per_batch[batch_num] : {}".format(max_mask_per_batch[batch_num]))
                                print("\nquotient : {}".format(quotient))
                                print("remainder : {}".format(remainder))                                                

                                print("\n마스킹할 수 있는 감정 토큰 후보 : {}".format(len(sentiment_candidate_for_mask)))
                                print("마스킹할 수 있는 일반 토큰 후보 : {}".format(len(common_candidate_for_mask)))                
                                print("마스킹하려는 감정 토큰 갯수 : {}/{}".format(max_senti_mask, masked_senti_token))
                                print("마스킹하려는 일반 토큰 갯수 : {}/{}".format(max_common_mask,masked_common_token))                                
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                            
                                break
                            else:   max_common_mask += 1

                        elif len(common_candidate_for_mask) < max_common_mask:
                            max_common_mask -= 1                            

                            if len(sentiment_candidate_for_mask) <= max_senti_mask:
                                self.total_masked_senti += max_senti_mask
                                self.total_masked_common += max_common_mask                            
                                break
                            else:                                
                                max_senti_mask += 1


                        elif (len(sentiment_candidate_for_mask) >= max_senti_mask
                            and len(common_candidate_for_mask) >= max_common_mask):
                            # 각 단어 유형별 마스킹 단어 추출     
                            self.total_masked_senti += max_senti_mask
                            self.total_masked_common += max_common_mask                            

                            if max_senti_mask > 0:          masked_type_labels.append(1)
                            elif max_senti_mask == 0:       masked_type_labels.append(2)
                            break
                
                if not self.args.weighted_masking_v1 and not self.args.weighted_masking_v2 and not self.args.weighted_masking_v3:
                    masked_senti_token  = random.sample(sentiment_candidate_for_mask, max_senti_mask)
                    masked_important_token = random.sample(important_candidate_for_mask, max_important_mask)                
                    masked_common_token = random.sample(common_candidate_for_mask, max_common_mask)

                # print("tokens num : {} / {}".format(len(masked_common_token) + len(masked_important_token) + len(masked_senti_token), max_mask_per_batch[batch_num]))

            else: # 마스킹된 토큰이 없는 Batch의 경우
                masked_type_labels.append(0)


            senti_indice        = [1 if position in masked_senti_token else 0 for position, token in enumerate(input_batch)]            
            important_indice    = [1 if position in masked_important_token else 0 for position, token in enumerate(input_batch)]            
            common_indice       = [1 if position in masked_common_token else 0 for position, token in enumerate(input_batch)]            

            # print("\nsenti_indice : {}/{}".format(type(senti_indice), senti_indice))
            # print("important_indice : {}/{}".format(type(important_indice), important_indice))
            # print("common_indice : {}/{}".format(type(common_indice), common_indice))      

            # print("senti_indice : {}".format(len([a for a in senti_indice if a == 1])))
            # print("common_indice : {}".format(len([a for a in common_indice if a == 1])))

            masked_sentiment_tokens.append(senti_indice.copy())
            masked_important_tokens.append(important_indice.copy())
            masked_common_tokens.append(common_indice.copy())

        # inputs : (batch, max_seq_length)
        # probability_matrix : (batch, max_seq_length)
        # probability_matrix[0] : [0.1500, 0.1500, ..., 0.1500]        
        probability_matrix = torch.full(inputs.shape, 0)        

        # special_tokens_mask : batch (length)
        # special_tokens_mask[0] : max_seq_length (length)
        # special_tokens_mask[0] : [1, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
        # CLS, SEP, PAD 토큰은 Masking하지 않기 위해 1로 표시        
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()]                

        
        # torch.bool 은 오류가 발생 -> uint8 사용 (최신 pytorch 버전에서는 bool이 사라진 듯)
        # print("masked_sentiment_tokens : {}/{}".format(type(masked_sentiment_tokens), masked_sentiment_tokens[0]))
        # print("masked_important_token : {}/{}".format(type(masked_important_token), masked_important_token[0]))
        # print("masked_common_tokens : {}/{}".format(type(masked_common_tokens), masked_common_tokens[0]))      

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)         # Special Token의 Masking 확률 0                        
        probability_matrix.masked_fill_(torch.tensor(masked_sentiment_tokens, dtype=torch.bool), value=1.0)     # Sentiment Token Masking       
        probability_matrix.masked_fill_(torch.tensor(masked_common_tokens, dtype=torch.bool), value=1.0)        # Common Token Masking
        if self.args.important_tokens:
            probability_matrix.masked_fill_(torch.tensor(masked_important_tokens, dtype=torch.bool), value=1.0)        # Common Token Masking        

        # masked_indices : (batch, max_seq_length)
        # 15% 확률을 바탕으로 1 or 0 예측 -> boolean type 변경        
        # masked_indices[0] : tensor([False, True, False, False, True, ..., False, False])
        # masked_indices 값 중, True는 Masking 대상
        probability_matrix = probability_matrix.float()
        masked_indices = torch.bernoulli(probability_matrix).type(torch.bool)

        return masked_indices, torch.tensor(masked_type_labels, dtype=torch.long)

    def gen_inputs_labels(self, inputs, masked_indices, masked_type_labels):
        original_inputs = inputs.clone()
        inputs = inputs.clone()
        labels = inputs.clone()

        # labels[0] : tensor([-100, -100, ..., 1056, -100, 234, -100, ..., -100])
        # Masking이 안된 Token들은 -100
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # print("\n\nA : ", torch.bernoulli(torch.full(labels.shape, 0.8)))        
        # print("B : ", torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool))
        # print("C : ", torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices)
        # print("D : ", masked_indices)

        masked_candiate = torch.full(labels.shape, 0.8)
        
        # [MASK] : 103
        indices_replaced = torch.bernoulli(masked_candiate).type(torch.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # get_polarity_labels(self, inputs, indices_senti_word,labels):
        # polarity_labels = self.get_polarity_labels(original_inputs,indices_senti_word,labels)
        # return inputs, labels, polarity_labels           

        return inputs, labels, masked_type_labels


    def get_polarity_labels(self, inputs, marked_senti_pos,labels):
        # Q. Masking된 일반 Token도 Word Polarity Prediction을 해야 하는가?

        # 1. 감성단어만 WPP를 수행하겠다
        # 1-1. 일반 Token은 -100으로 Labels 설정
        polarity_labels = torch.full(inputs.shape, -100, dtype=torch.long)

        # 1-2. 감정 단어는 저장된 WP로 레이블 갱신
        boolean_senti_word = torch.tensor(marked_senti_pos, dtype=torch.bool)
        for batch_num in range(len(boolean_senti_word)):
            for idx in range(len(boolean_senti_word[batch_num])):
                if boolean_senti_word[batch_num][idx] == True:
                    polarity_labels[batch_num][idx] = self.senti_words[inputs[batch_num][idx].item()][1]

        # 2. 마스킹된 일반 단어에 NEU 레이블링
        for batch_num in range(len(polarity_labels)):
            for idx in range(len(polarity_labels[batch_num])):
                # Masked Label인데 감성 단어로는 레이블링이 안되어 있는 경우 -> 마스킹된 일반 토큰
                if not labels[batch_num][idx] == -100 and polarity_labels[batch_num][idx] == -100:
                    # NEG : 0
                    # POS : 1
                    # NEU : 2
                    polarity_labels[batch_num][idx] = 2    
        print("polarity_labels : {}".format(polarity_labels))
        return polarity_labels  
