from util import *
import nltk


def _post_step(outputs):
    pass

def train(args, model, train_dataloader, dev_dataloader=None, train_masker=None, dev_masker=None):

    t_total = len(train_dataloader)//args.gradient_accumulation_steps * args.epochs    
    no_decay = {"bias","LayerNorm.weight"}
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    # https://huggingface.co/roberta-large/blob/main/README.md
    # 위 링크 참조해서 파라미터 설정
    optimizer = AdamW(params = optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(t_total * args.warmup_proportion), num_training_steps = t_total)    

    # Check if saved optimizer or scheduler states exist
    if (args.additional_learning and args.model_name_or_path is not None 
        and os.path.isfile(os.path.join(args.model_name_or_path, "my_checkpoint.pth.tar"))):
        # Load in optimizer and scheduler states
        print("\n학습된 모델 있음! {}\n".format(args.model_name_or_path))
        if args.additional_learning:    model, optimizer, scheduler = load_checkpoint(torch.load(os.path.join(args.model_name_or_path, "my_checkpoint.pth.tar")), model, optimizer, scheduler)
        else:                           model, _, _ = load_checkpoint(torch.load(os.path.join(args.model_name_or_path, "my_checkpoint.pth.tar")), model)


    print("\nStart training...\n")
    if args.mode == 'fine': print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^14} | {'Elapsed':^9}")
    else: print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Perplexity':^14} | {'Elapsed':^9}")
    print("-"*60)

    # 학습 추이를 텍스트파일로 저장 (지워도 됨)
    save_ = open("./{}dev_set_result.txt".format(args.seed), mode="w", encoding="UTF-8-sig", errors="ignore")
    save_.write("Epoch\t\tTrain_loss\t\tVal_loss\t\tACC(Perplexity)\n")
    save_.close()
    val_acc = best_acc = 0
    best_epoch = 0
    best_model = best_optimizer = best_scheduler = None
    perplexity = best_perflexity = best_loss = float("inf")

    model.zero_grad()            

    dev_acc  = list()
    dev_loss = list()
    # 학습 시작 (Epoch 단위)
    for epoch_i in trange(args.epochs, leave=True, position=0):
        t0_epoch = time.time()

        model.train()            
        # Train
        total_loss = 0
        masked_token_num = {'total_tokens':0 , 'sentiment_tokens':0, 'common_tokens':0, 'important_tokens':0}
        args.mtl = 1
        task_A = [0] * math.floor(len(train_dataloader)/2)
        task_B = [1] * (math.floor(len(train_dataloader)/2) + (len(train_dataloader)%2))
        task = task_A + task_B
        random.shuffle(task)
        for step, (batch,task_num) in enumerate(tqdm(zip(train_dataloader,task), total=len(train_dataloader), leave=True, position=0)):        
            inputs, masked_token_num = train_set(args=args,epoch=epoch_i+1, batch=batch, train_masker=train_masker,masked_token_num=masked_token_num, train=True)
            args.mtl = task_num
            # Mask 토큰이 없는 경우 터진다
            try:    
                outputs = model(**inputs)
                del inputs
            except Exception as e:  
                print("Error: {0}".format(e))
                continue
                
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps



            total_loss += loss.detach().cpu().numpy().item()        
            loss.backward()
            del loss
            if args.mode == 'post' or args.mode== 'fine' and args.mlm: _post_step(outputs)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()                            
                optimizer.zero_grad()                    
                torch.cuda.empty_cache()
        if args.mlm:             
            # print("args.total_tokens : {} {} ".format(type(args.total_tokens), args.total_tokens))
            # print("masked_token_num['total_tokens'] : {} {}".format(type(masked_token_num['total_tokens']), masked_token_num['total_tokens']))            
            print("\n{0} Epoch\t: {1} masked ({2}%): ".format(epoch_i+1, masked_token_num['total_tokens'], round(masked_token_num['total_tokens']/args.total_tokens*100,2)))
            print("Sentiment Tokens\t: {0} masked ({1}%): ".format(masked_token_num['sentiment_tokens'], round(masked_token_num['sentiment_tokens']/masked_token_num['total_tokens']*100,2)))        
            print("Important Tokens\t: {0} masked ({1}%): ".format(masked_token_num['important_tokens'], round(masked_token_num['important_tokens']/masked_token_num['total_tokens']*100,2)))                    
            print("Common Tokens\t: {0} masked ({1}%): ".format(masked_token_num['common_tokens'], round(masked_token_num['common_tokens']/masked_token_num['total_tokens']*100,2)))        

        train_loss = total_loss / len(train_dataloader)         


        # 학습한 모델 Validation set으로 성능 평가
        if dev_dataloader is not None: 
            if args.mode == 'fine':     val_loss, val_acc = evaluate(args=args, epoch= epoch_i +1, model=model, dev_dataloader=dev_dataloader, dev_masker =dev_masker)
            elif args.mode == 'post':   val_loss, val_perplexity = evaluate(args=args, epoch = epoch_i+1, model=model, dev_dataloader=dev_dataloader, dev_masker=dev_masker)

            # 성능 평가 후, 모델 저장
            if (args.mode == 'fine' 
                and (val_acc > best_acc 
                    or (val_acc == best_acc and val_loss <= best_loss))):
                best_loss = val_loss                
                best_acc = val_acc                
                best_epoch = epoch_i + 1
                best_model = copy.deepcopy(model)
                save_checkpoint(args, best_epoch, best_model, optimizer, scheduler, best_acc)

            elif (args.mode == 'post'
                and (val_perplexity < best_perflexity
                    or (val_perplexity == best_perflexity and val_loss <= best_loss))):
                best_loss = val_loss                
                best_perflexity = val_perplexity           
                best_epoch = epoch_i + 1
                best_model = copy.deepcopy(model)
                save_checkpoint(args, best_epoch, best_model, optimizer, scheduler, best_perflexity)

            time_elapsed = time.time() - t0_epoch

            # 성능 평가 결과 Print/Write
            if (epoch_i+1) % 1 == 0:
                save_ = open("./{}dev_set_result.txt".format(args.seed), mode="a", encoding="UTF-8-sig", errors="ignore")
                if args.mode == 'fine': 
                    print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {val_acc:^14.5f} | {time_elapsed:^9.2f}")
                    save_.write("{0}\t\t{1}\t\t{2}\t\t{3}\n".format(epoch_i,round(train_loss,5),val_loss,val_acc))
                    dev_acc.append(str(val_acc))
                    dev_loss.append(str(val_loss))
                else: 
                    print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {val_perplexity  :^14.5f} | {time_elapsed:^9.2f}")
                    save_.write("{0}\t\t{1}\t\t{2}\t\t{3}\n".format(epoch_i,round(train_loss,5),round(val_loss,5),round(val_perplexity,5)))
                save_.close()

    print("\nTraining complete!")
    print(f"Best model epoch: {best_epoch}") 
    save_ = open("./{}dev_set_result.txt".format(args.seed), mode="a", encoding="UTF-8-sig", errors="ignore")
    acc = ("\t").join(dev_acc) + "\n"
    loss = ("\t").join(dev_loss) + "\n"
    save_.write("Dev Acc : {}".format(acc))
    save_.write("Dev Loss : {}".format(loss))
    save_.close()
    return  best_model
    



def evaluate(args=None, epoch=0, model=None, dev_dataloader=None, dev_masker=None, val_test='val', original_inptus=None, num_labels=0):
    if val_test == 'test' and not args.reformulate_entailment:
        # Check if saved optimizer or scheduler states exist
        if args.model_name_or_path is not None and os.path.isfile(os.path.join(args.model_name_or_path, "best_model.bin")):
        # if args.model_name_or_path is not None and os.path.isfile(os.path.join(args.model_name_or_path, "best_model.bin")):
            print("테스트를 위한 모델 변경")
            model_name = args.model.split('-')[0].lower()
            model_name = "{}_mlm".format(model_name) if args.mlm else model_name
            model_class, tokenizer_class = MODEL_CLASS[model_name]            
            config = AutoConfig.from_pretrained(args.model)
            config.num_labels = num_labels
            saved_model = args.model_name_or_path + "/best_model.bin"
            model = model_class.from_pretrained(pretrained_model_name_or_path=saved_model, config=config)
            model.to(args.device)
            # Load in optimizer and scheduler states
            # model, _, _ = load_checkpoint(checkpoint=torch.load(os.path.join(args.model_name_or_path, "my_checkpoint.pth.tar")), model=model)

    model.eval()      
    preds_list = list()
    labels_list = list()    
    # val_accuracy = list()
    val_accuracy = 0    
    eval_loss = 0        


    task_A = [0] * math.floor(len(dev_dataloader)/2)
    task_B = [1] * (math.floor(len(dev_dataloader)/2) + (len(dev_dataloader)%2))
    task = task_A + task_B
    random.shuffle(task)

    original_score = list()
    for step, (batch, task_num) in enumerate(tqdm(zip(dev_dataloader,task), total=len(dev_dataloader), desc="Evaluate : ", leave=True, position=0)):
        inputs, masked_token_num = train_set(args=args,batch=batch, train_masker=dev_masker)
        args.mtl = task_num

        if val_test =='test':
            search_space = list(inputs.keys())
            for key in search_space:
                if not key in ['input_ids', 'attention_mask', 'labels']:
                    del inputs[key]

        with torch.no_grad():
            # Mask 토큰이 없는 경우 터진다
            try:    
                outputs = model(**inputs)
                loss = outputs[0]
                eval_loss += loss.mean().item()
            except Exception as e:
                print("Error: {0}".format(e))
                continue


        if args.mode == 'fine':
            preds = torch.argmax(outputs[1], dim=1).flatten()
            preds_list.extend(p.item() for p in preds)            
            labels_list.extend(l.item() for l in inputs['labels'])
            # accuracy = (preds == inputs['labels']).cpu().numpy().mean() * 100
            # accuracy = accuracy_score(labels_list, preds_list) * 100
            # val_accuracy.append(accuracy)     
        del inputs

    eval_loss = eval_loss/(step+1)

    if args.mode == 'fine':
        val_accuracy = accuracy_score(labels_list, preds_list) * 100        

        if val_test == 'test':              
            print("\nVAL TEST ACC : {}".format(val_accuracy))
            print("\naccuracy_score \t:", val_accuracy)
            print("Precision \t:", precision_score(labels_list, preds_list, average='macro') * 100)
            print("Recall \t:", recall_score(labels_list, preds_list, average='macro') * 100)
            print("F1 \t:", f1_score(labels_list, preds_list, average='macro') * 100, "\n")

            save_ = open("./{}test_set_result.txt".format(args.seed), mode="w", encoding="UTF-8-sig", errors="ignore")
            save_.write("Accuracy\t:\t" + str(accuracy_score(labels_list, preds_list) * 100) + "\n")
            save_.write("Precision\t:\t" + str(precision_score(labels_list, preds_list, average='macro') * 100) + "\n")
            save_.write("Recall\t:\t" + str(recall_score(labels_list, preds_list, average='macro') * 100) + "\n")            
            save_.write("F1 Score\t:\t" + str(f1_score(labels_list, preds_list, average='macro') * 100) + "\n")                

            save_.write("\n[ 틀린 데이터 ]\n")                
            for idx, (l, p) in enumerate(zip(labels_list, preds_list)):
                if not l == p:
                    save_.write("{0} 모델 : {1} / 정답 : {2} / 예제 : {3}\n\n".format(idx, p,l, original_inptus[idx]))
            save_.close()


        return eval_loss, val_accuracy
        
    elif args.mode == 'post':
        perplexity = torch.exp(torch.tensor(eval_loss))
        return eval_loss, perplexity.item()


def compute_score(args=None, model=None, dataloader=None, labels=None):
    model.eval()      

    preds_scores = list()
    softmax = nn.Softmax(dim=1)
    for step, batch in enumerate(tqdm(dataloader, desc="Compute Original Score : ", leave=True, position=0)):
        inputs, masked_token_num = train_set(args=args,epoch=None, batch=batch, train_masker=None)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = softmax(outputs[1])
            preds_scores.extend(logits.detach().cpu().numpy().tolist())
    # preds_score : [6,920 * 2]    
    save_scores = [score[label] for score, label in zip(preds_scores,labels)]

    # print("save_scores : {}".format(save_scores))
    # print("\nOriginal Score Avg. : {}".format(round(sum(save_scores)/len(save_scores),3)))
    with gzip.open("./data/scores/original_SST2.score", mode='w') as out:
        pickle.dump(save_scores, out)
    print("Original Score {}개 저장 완료".format(len(save_scores)))


def compare_scores_and_extract_tokens_with_mask(args=None, model=None, data=None, labels=None):

    model.eval()      
    softmax = nn.Softmax(dim=1)

    important_tokens = dict()

    # 01. Load Original Sentence Score
    with gzip.open("./data/scores/original_SST2.score", mode='r') as inp:
        original_score = pickle.load(inp)


    # 02. Define Tokenizer
    model_name = args.model.split('-')[0].lower()
    model_name = "{}_mlm".format(model_name) if args.mlm else model_name
    model_class, tokenizer_class = MODEL_CLASS[model_name]            

    tokenizer = tokenizer_class.from_pretrained(args.model)

    # 03. Tokenize
    tokenized_data = [tokenizer.tokenize(d) for d in data]


    cls_token = tokenizer.cls_token if not 'robert' in args.model else tokenizer.bos_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token 

    # a = list()
    # b = list()

    print("Len tokenized_data : {}".format(len(tokenized_data)))
    print("Len original score : {}".format(len(original_score)))
    for num, (compare_data, original, label) in enumerate(tqdm(zip(tokenized_data, original_score, labels), total=len(data), desc="Compare Score : ", leave=True, position=0)):        
        # print("\nTF-IDF : {}".format(tfidf))        
        # 04. 한 입력 데이터로부터 Token을 하나씩 제거하며 연산
        # compare_data = ['i','love','you']

        seq_buffer = list()
        pos_buffer = list()
        for pos in range(len(compare_data)):
            tmp_data = compare_data.copy()

            # If threshold == 0.1, 중요한 단어는 마스킹에서 빼겠다.
            ### 아래의 과정이 마스킹하는 과정
            tmp_data[pos] = mask_token          # 순차적으로 한 token 마스킹
            seq_buffer.append(tmp_data.copy())  # masked 입력 데이터를 추가
            pos_buffer.append(pos)
            ###
            tmp_data.clear()

        new_seq_buffer = list()
        for tokens in seq_buffer:
            if len(tokens) >= args.max_seq_length-1:    tokens = tokens[:args.max_seq_length-2]                       
            tokens = [cls_token] + tokens + [sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)        
            input_mask = [1] * len(input_ids)
            padding_length = args.max_seq_length - len(input_ids)

            input_ids += [tokenizer.convert_tokens_to_ids(pad_token)] * padding_length           
            input_mask += [0] * padding_length

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length            
            new_seq_buffer.append({'input_ids' : input_ids.copy(), 'input_mask': input_mask.copy()})
        inputs_ids_tensor = torch.tensor([data['input_ids'] for data in new_seq_buffer], dtype=torch.long)
        inputs_mask_tensor = torch.tensor([data['input_mask'] for data in new_seq_buffer], dtype=torch.long)


        label_list      = [label] * len(pos_buffer)
        label_tensor    = torch.tensor(label_list, dtype=torch.long)
        pos_tensor      = torch.tensor(pos_buffer, dtype=torch.long)

        # print("inputs_ids_tensor : {}".format(inputs_ids_tensor.size()))
        # print("inputs_mask_tensor : {}".format(inputs_mask_tensor.size()))
        # print("label_tensor : {}".format(label_tensor.size()))
        # print("pos_tensor : {}".format(pos_tensor.size()))
        
        tensor_data  = TensorDataset(inputs_ids_tensor, inputs_mask_tensor, label_tensor,pos_tensor)
        dataloader   = DataLoader(tensor_data, batch_size=200, shuffle=False)

        
        tokens_len = [len(batch[1]) for btach in dataloader]
        avg_len = sum(tokens_len) / len(tokens_len)
        for batch in dataloader:  

            ### 각 배치별 토큰 갯수 ###
            tokens_len = len(batch[1]) * [0]   

            count_len = torch.nonzero(batch[1] == 1).detach().cpu().numpy().tolist()
            for batch_num, _ in count_len:  tokens_len[batch_num] += 1

            inputs = {'input_ids' : batch[0].to(args.device),\
                    'attention_mask':batch[1].to(args.device),\
                    'labels' : batch[2].to(args.device)}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = softmax(outputs[1])
                del inputs
            logits = logits.detach().cpu().numpy().tolist()
            pos_b = batch[3].detach().cpu().numpy().tolist()

            # POS는 한 입력 내 각 토큰 위치라고 봐도 무방
            for batch_num, (score,pos) in enumerate(zip(logits, pos_b)):
                cur_token = compare_data[pos]


                # 평균으로 구하는 방식
                # if not cur_token in important_tokens:
                #     important_tokens[cur_token] = [tokenizer.convert_tokens_to_ids(cur_token),0, original,0]
                
                # important_tokens[cur_token][1] += 1
                # important_tokens[cur_token][3] = (important_tokens[cur_token][3] + score[label])/2
                if not cur_token in important_tokens:
                    important_tokens[cur_token] = 0
                
                cur_score = abs(original - score[label])
                # print("\noriginal : {} / score : {}".format(original, score[label]))
                # print("cur_score : {} / normalization : {}".format(cur_score, math.log10(tokens_len[batch_num])))

                if args.length_normalization:   
                    cur_score *= math.log10(tokens_len[batch_num])
                # print("length_normalization_score : {}".format(cur_score))

                # score[label] -> 모델이 True Label에 대해 예측한 값 -> 이걸 우리는 Importance score로 볼 것

                important_tokens[cur_token] = (important_tokens[cur_token] + cur_score)/2

            # tf_idf 조건문 때문에 빠진 단어들은 점수를 기록해야한다
            # undefined_tokens = [pos for pos in range(len(compare_data)) if not pos in pos_b]
            # for pos in undefined_tokens:
            #     cur_token = compare_data[pos]
            #     if not cur_token in important_tokens:
            #         important_tokens[cur_token] = 1

    with gzip.open("./data/scores/(SST)Token_important_scores_with_length_normalization_ln_2021_11_07.score", mode='w') as out:
        pickle.dump(important_tokens, out)
    print("Important Tokens {}개 저장 완료".format(len(important_tokens.keys())))

def compute_tf_idf(tokenized_data):
    print("안뇽")
    print("tokenized_data : {}".format(tokenized_data))
    # TF
    data = [(" ").join(inp) for inp in tokenized_data]
    # vectorizer = TfidfVectorizer(smooth_idf=True, stop_words='english')
    vectorizer = TfidfVectorizer(smooth_idf=True,stop_words=None, token_pattern=r'\S+')

    X = vectorizer.fit_transform(data)
    scores = dict()
    for num in trange(X.shape[0], desc="Compute tfidf"):        
        for key, value in dict(zip(vectorizer.get_feature_names(), X.toarray()[num])).items():

            if not value == 0.0:
                key = int(key)
                if not key in scores:
                    scores[key] = value
                else:
                    scores[key] = (scores.get(key) + value)/2

    print("scores : {}".format(scores))
    with gzip.open("./data/scores/(SST)TF_IDF_2021_12_27.score", mode='w') as out:
        pickle.dump(scores, out)

if __name__ == "__main__":
    # 1 모델에서 사용할 하이퍼 파라미터를 선언
    parser = default_parser(argparse.ArgumentParser())
    args = parser.parse_args()

    print("===== [ 모델 학습 정보 ({}) ] =====".format(time.time()))
    print("MLM Loss : {}".format(args.mlm_loss))
    # Random Seed 선언
    seed_everything(args.seed)

    # CPU/GPU 설정
    set_device(args)

    # 2 데이터를 불러오기
    data, labels= load_data(args)

#     data['train'] = data['train'][:40]
#     labels['train'] = labels['train'][:40]    

#     data['dev'] = data['dev'][:40]
#     labels['dev'] = labels['dev'][:40]    

#     data['test'] = data['test'][:20]
#     labels['test'] = labels['test'][:20]    


    # 3 데이터 전처리
    model, tokenized_data, masker, tokens_list = preprocess(args, data,labels)
    
    # model -> GPU로 옮기는 것
    model.to(args.device)        


    # 4 전처리된 데이터 -> Tensor
    train_dataloader, dev_dataloader, test_dataloader = idx_to_tensor(args=args, tokenized_data= tokenized_data,labels=labels, batch_size=args.batch_size)        

    # 5. 학습 시작
    if args.train:
        model = train(args, model, train_dataloader, dev_dataloader,train_masker=masker['train'], dev_masker=masker['eval'])

    if args.eval:
        num_labels = len(set(labels['train'] + labels['dev']+labels['test']))                
        _ , acc_perplexity = evaluate(args, model=model , dev_dataloader=test_dataloader, dev_masker=masker['eval'], val_test='test', original_inptus=data['test'], num_labels=num_labels)

    if args.compute_score:
        compute_score(args=args, model=model, dataloader=train_dataloader, labels=labels['train'])

    if args.extract_important_tokens:
        compare_scores_and_extract_tokens_with_mask(args=args, model=model, data=data['train'], labels=labels['train'])

    if args.compute_tfidf:
        compute_tf_idf(tokens_list)
        print("계산 완료")