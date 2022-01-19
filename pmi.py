import math
import nltk
import numpy as np
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer
from nltk import pos_tag
from tqdm import tqdm
from string import punctuation
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# 20210330
# 1. like 처럼 POS/NEG 양쪽에 들어간 단어가 있다.
# 2. f와 h처럼, 한 글자 짜리도 감정 단어로 들어간다.  -> stem_dict에 길이 1짜리는 반영하지 않는다.
# 3. '' / "" 같은 것들이 감정 단어로 분류된다.        -> punctuation을 지우는게 정답인가?

# 20210402
# Stemmer의 문제점은 모든 문자는 소문자로 만든다는 것이다.
# 이에 따라 Corpus에서 대문자로 나온 문자는 PMI 연산 과정에서 소문자로 바뀌고, 최종적으로는 UNK가 사전에 들어가게 된다.
class Agument_PMI():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stemmer = PorterStemmer()
        self.speical_tokens = ['</s>','<s>','<unk>','<pad>','<mask>']
        self.stem_dict = dict()
        self.word2idx = dict()
        self.idx2word = dict()
        self.avg_co_count = {'num': 0 , 'avg': 0}
        self.my_punctuation = punctuation.replace("!", "")
        self.my_punctuation = self.my_punctuation.replace("?", "")

    def remove_punctuation(self,text):        
        return text.translate(str.maketrans("", "", self.my_punctuation))

    def remove_number(self,text):        
        numbers = "0123456789"
        return text.translate(str.maketrans("", "", numbers))

    def load_seed_words(self, polarity):
        with open("./data/lexicon/" + polarity + "_words.txt", mode="r", encoding="UTF-8-sig", errors="ignore") as inp: 
            words = inp.read().splitlines()   
            senti_words = list()
            for w in words:
                stem_token = self.stemmer.stem(w.lower())

                if not stem_token in self.stem_dict.keys():
                    self.stem_dict[stem_token] = list()

                self.stem_dict[stem_token].append(w)
                self.stem_dict[stem_token] = list(set(self.stem_dict.get(stem_token)))                        

                senti_words.append(stem_token)
            senti_words = list(set(senti_words))
            return senti_words.copy()

    def clean_corpus(self,corpus):
        # 1-1. Add Stop words
        clean_candiate = set(stopwords.words('english')) 

        # # 1-2. Add Speical Token
        # clean_candiate.update(self.speical_tokens)

        # 1-3. Stemmer
        clean_candiate = list(set([self.stemmer.stem(token.lower()) for token in clean_candiate]))

        # 1-4. Remove Seed word from candidate
        clean_candiate = [token for token in clean_candiate if not token in self.pos_seed_words + self.neg_seed_words]

        # 2. Clean corpus
        before_num = 0
        after_num = 0        
        new_corpus = list()
        new_seq = list()        
        for seq in tqdm(corpus, leave=True, position=0):
            before_num += len(seq)
            # print(pos_tag(seq))
            for token in seq:
                new_token = token
                # 2-1. Remove Roberta Symbol Ġ
                # Roberta는 띄어쓰기 단위마다 Ġ 심볼이 들어간다
                if token[0] == 'Ġ': new_token = new_token[1:]

                # 2-2 Remove Punctuation
                new_token = self.remove_punctuation(new_token)
                new_token = self.remove_number(new_token)

                # 2-3. 두 글자이하인 토큰은 제거
                if len(new_token) <= 2: continue

                # 2-4. Lower
                # Roberta는 case 모델이므로, 이 과정이 필요하다
                new_token = new_token.lower()

                # Candidate     : ['love', 'loved', 'loves', 'lovely', 'he', 'his', 'him', 'be', 'is', 'are', 'were']
                # Stemmer       : ['love', 'love',  'love',  'love',   'he', 'hi',  'him', 'be', 'is', 'are', 'were']
                # Lemmatizer    : ['love', 'loved', 'love',  'lovely', 'he', 'his', 'him', 'be', 'is', 'are', 'were']
                # --> Stemmer를 써야 과거형, 복수형을 잡을 수 있다.
                stem_token = self.stemmer.stem(new_token)

                # 2-5. Remove cadidate from corpus:
                if not stem_token in clean_candiate:
                    if not stem_token in self.stem_dict.keys(): self.stem_dict[stem_token] = list()
                    self.stem_dict[stem_token].append(token)
                    self.stem_dict[stem_token] = list(set(self.stem_dict.get(stem_token)))                        
                    new_seq.append(stem_token)

            # 정제된 토큰만으로 구성된 Corpus
            # 이 Corpus로 Agumentation할 토큰을 찾을 것
            after_num += len(new_seq)
            new_corpus.append(new_seq.copy())
            new_seq.clear()
        # print("\n기존 Corpus\t: {0} ( Token 수 )".format(before_num))
        # print("정제된 Corpus\t: {0} ( Token 수 )".format(after_num))
        return new_corpus.copy()

    def make_vocab(self,corpus):
        self.idx2word = list(sorted(list(set([token for seq in corpus for token in seq]))))      
        for word in self.idx2word:   self.word2idx[word] = len(self.word2idx)

    def make_cooccurrences_matrix(self,corpus):
        co_occurrences_pair = dict()
        for tokens in tqdm(corpus, leave=True, position=0):
            # 각 문장마다 한 번씩만 빈도수를 세므로, 중복 제거
            tokens = set(tokens)

            # 오름차순 정렬
            # matrix(A,B)와 matrix(B,A)는 동시 등장 횟수를 동일하게 세야한다. 
            # 따라서 항상 가나다 순으로 정렬해서 동시 등장 횟수 카운트
            tokens = list(sorted([self.word2idx[token] for token in tokens]))
            for t_stamp, pivot_token in enumerate(tokens):
                # 자기 자신부터 카운트하는 이유는 WP를 계산할 때, 자기 자신이 등장한 빈도수를 고려하기 때문
                for compare_token in tokens[t_stamp:]:
                    co_occurrences_pair[pivot_token,compare_token] = co_occurrences_pair.get((pivot_token,compare_token), 0) + 1

        co_occurrences_matrix = np.zeros((len(self.word2idx), len(self.word2idx)),dtype=int)  


        for (key1, key2) in co_occurrences_pair.keys():
            co_occurrences_matrix[key1][key2] = co_occurrences_pair[key1,key2]
            co_occurrences_matrix[key2][key1] = co_occurrences_pair[key1,key2]

        return co_occurrences_matrix.copy()         


    def calculate_wp(self,co_occurrences_matrix,seed_word,word,corpus_len):
        try:
            # frequency of token a
            a = co_occurrences_matrix[seed_word][seed_word]

            # frequency of token b
            b = co_occurrences_matrix[word][word]

            # co-occurrence of token a and b
            a_b = co_occurrences_matrix[seed_word][word]
            self.avg_co_count['avg'] += a_b
            self.avg_co_count['num'] += 1

            # 1. 함께 등장한 적 없는 경우(0)
            if a_b == 0:    return 0
            prob_a_b = a_b/corpus_len
            prob_a = a / corpus_len 
            prob_b = b / corpus_len
            return np.log(prob_a_b/prob_a*prob_b)

        except Exception as e:
            # 두 가지 경우에 에러가 발생한다.
            # 첫 번째, Seed word가 해당 코퍼스에 등장하지 않는 경우
            # 두 번째, Seed word와 해당 word가 함께 등장한 적 없는 경우
            # -> WP를 0으로 반환
            print("실패 : ", e)
            return 0


    def run_pmi(self, args, corpus):
        # 1. Load Seed words
        self.pos_seed_words = self.load_seed_words(polarity="pos")
        self.neg_seed_words = self.load_seed_words(polarity="neg")

        # 2. Clean Corpus
        c_corpus = self.clean_corpus(corpus=corpus)

        # 3. Make Vocab
        self.make_vocab(c_corpus)

        # 4. Count co-occurrence
        co_occurrences_matrix = self.make_cooccurrences_matrix(c_corpus)
        # 동시 등장 횟수가 적은 경우 matrix로 만들지 않는다.
        # 즉 해당값은 0으로 계산되는 것
        # 이렇게 하는 이유는 seed word를 기준으로 너무 많은 감성단어가 파생되기 때문

        # Corpus에 없는 Seed 단어가 있다
        # 예를들어 Roberta에서 'Ġlove'는 나오지만, 'love'가 나오지 않는다면, 이는 문제 발생
        # 실제 Corpus에 등장하지 않는 Seed 단어는 제거 필요    
        # pos_seed_words = [word for word in pos_seed_words if word in word2idx.keys()]
        # neg_seed_words = [word for word in neg_seed_words if word in word2idx.keys()]

        word_polarity = {word:0 for word in self.word2idx.keys()}
        corpus_len = len(corpus)

        for word in tqdm(word_polarity.keys(), desc="pmi - word polarity calculate", leave=True, position=0):   
            for seed_word in self.pos_seed_words:
                if not seed_word in self.word2idx.keys():   continue
                word_polarity[word] += self.calculate_wp(co_occurrences_matrix,self.word2idx[seed_word],self.word2idx[word],corpus_len)
            for seed_word in self.neg_seed_words:        
                if not seed_word in self.word2idx.keys():   continue
                word_polarity[word] -= self.calculate_wp(co_occurrences_matrix,self.word2idx[seed_word],self.word2idx[word],corpus_len)



        # 음수 - 음수 = '양수' -> Neg에 가까운 단어.
        # 음수 - 음수 = '음수' -> Pos에 가까운 단어.
        pos_words_list = dict()
        neg_words_list = dict()

        # 각 극성마다, 평균 점수 넘는 단어들만 추출
        for key,value in word_polarity.items():
            if not value == 0:
                # 만약 특정 시드 단어가 반대 감성에서 자주 나오는 경우
                # 실제로 like가 부정 단어와 많이 사용되어, negative 단어로 분류되어버림
                # Seed 단어라면, 원래 레이블을 따라간다.

                if key in self.pos_seed_words:          pos_words_list[key] = 10000
                elif key in self.neg_seed_words:        neg_words_list[key] = 10000

                if value > 0:   pos_words_list[key] = abs(value)
                elif value < 0: neg_words_list[key] = abs(value)

        # 각 감성 별로 상위 일정 % 단어가 뽑혀야 균형이 맞춰진다.
        # 감정 단어 마스킹은 감정의 극성을 사용하지 않지만, 일반적으로 부정 단어가 더 많다는 점을 볼 때 너무 한 극성에 편향될 수 있다.
        sorted_pos_words = list(sorted(pos_words_list.items(), key=(lambda x: x[1]), reverse = True))    # 큰 양수가 앞으로
        sorted_neg_words = list(sorted(neg_words_list.items(),  key=(lambda x: x[1]), reverse = True))   # 큰 음수가 앞으로

        # 이미 정렬을 했으므로, 더이상 value 값은 중요하지 않다.
        selected_pos_words = [token_ids for num, (token_ids, value) in enumerate(sorted_pos_words) if int(args.pmi * len(sorted_pos_words)) > num]
        selected_neg_words = [token_ids for num, (token_ids, value) in enumerate(sorted_neg_words) if int(args.pmi * len(sorted_neg_words)) > num]
        
        # print("\nPOS : {}개".format(len(selected_pos_words)))
        # print("NEG : {}개".format(len(selected_neg_words)))

        senti_words = dict()
        for stem_token in selected_pos_words:
            for original_token in self.stem_dict[stem_token]:
                # 4 case
                # (1) love
                # (2) Love
                # (3) Ġlove
                # (4) ĠLove
                case_1 = ""
                case_2 = ""
                case_3 = ""
                case_4 = ""
                if original_token[0] == 'Ġ':
                    case_1 = original_token[1].lower() + original_token[2:]
                    case_2 = original_token[1].upper() + original_token[2:]                    
                    case_3 = original_token[0] + original_token[1].lower() + original_token[2:]                    
                    case_4 = original_token[0] + original_token[1].upper() + original_token[2:]
                else:
                    case_1 = original_token[0].lower() + original_token[1:]
                    case_2 = original_token[0].upper() + original_token[1:]
                    case_3 = 'Ġ' + original_token[0].lower() + original_token[1:]
                    case_4 = 'Ġ' + original_token[0].upper() + original_token[1:]
                cases = list(set([case_1, case_2, case_3, case_4]))
                for case in cases:
                    ids = self.tokenizer.convert_tokens_to_ids(case)
                    if ids == 3:    # <unk> token
                        continue
                    # convert_tokens_to_ids의 결과, vocab에 없는 단어라서 
                    if stem_token in self.pos_seed_words:     senti_words[ids] = (ids, 2, 1000)
                    else:                                     senti_words[ids] = (ids, 2, abs(pos_words_list[stem_token]))

        pos = len(senti_words)
        print("\nPos 최종 단어 갯수 : {0}".format(len(senti_words)))

        for stem_token in selected_neg_words:
            for original_token in self.stem_dict[stem_token]:
                # 4 case
                # (1) love
                # (2) Love
                # (3) Ġlove
                # (4) ĠLove
                case_1 = ""
                case_2 = ""
                case_3 = ""
                case_4 = ""                
                if original_token[0] == 'Ġ':
                    # 단어 길이가 2이하인 경우, 의미있는 단어라 보지 않는다.
                    case_1 = original_token[1].lower() + original_token[2:]
                    case_2 = original_token[1].upper() + original_token[2:]                    
                    case_3 = original_token[0] + original_token[1].lower() + original_token[2:]                    
                    case_4 = original_token[0] + original_token[1].upper() + original_token[2:]
                else:
                    # 단어 길이가 2이하인 경우, 의미있는 단어라 보지 않는다.
                    case_1 = original_token[0].lower() + original_token[1:]
                    case_2 = original_token[0].upper() + original_token[1:]
                    case_3 = 'Ġ' + original_token[0].lower() + original_token[1:]
                    case_4 = 'Ġ' + original_token[0].upper() + original_token[1:]
                   
                cases = list(set([case_1, case_2, case_3, case_4]))
                for case in cases:
                    ids = self.tokenizer.convert_tokens_to_ids(case)
                    if ids == 3:    # <unk> token
                        continue
                    # convert_tokens_to_ids의 결과, vocab에 없는 단어라서 
                    if stem_token in self.neg_seed_words:     senti_words[ids] = (ids, 1, 1000)
                    else:                                     senti_words[ids] = (ids, 1, abs(neg_words_list[stem_token]))

        print("Neg 최종 단어 갯수 : {0}".format(len(senti_words)-pos))
        print("총 감정단어 : {0}".format(len(senti_words)))
        # print("예시 : ",senti_words[49952])


        # print([self.tokenizer.convert_ids_to_tokens(ids) for ids in senti_words.keys()])
        return senti_words
     