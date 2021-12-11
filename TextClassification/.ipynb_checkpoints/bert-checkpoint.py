# import tensorflow as tf
import torch
from transformers import BertTokenizer
# from transformers import BertForSequenceClassification, AdamW, BertConfig
# from transformers import get_linear_schedule_with_warmup
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# import pandas as pd
import numpy as np
# import random
# import time

''' 구현해야 할 것
1. pretrained model 불러와서 model로 우리 데이터의 단어/문장들에 대해서 classification하는 함수
2. classification 결과를 input_test에 새로운 key로 저장하는 함수
3. key값이 0인 단어/문장들을 삭제하고, key를 삭제하는 함수
4. 최종 결과를 ELK로 보내는 함수
'''

def convert_input_data(sentences):

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

def BertModel(sentences, PATH):

    ''' 
    [Input]
    - sentence : test data
    - PATH : location of pretrained model
    [Output]
    - logits : test result (0: non-ad, 1: ad)
    '''

    # select device to run model
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    saved_model = torch.load(PATH) # bring pretrained model
    saved_model.eval()

    inputs, masks = convert_input_data(sentences) # convert sentence to input data
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device) # pass data to gpu
    with torch.no_grad():     
        outputs = saved_model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0] # bring loss
    logits = logits.detach().cpu().numpy() # move data to cpu
    return np.argmax(logits[0]) # return 0 or 1

def DropAd(input_dict):
    sentences = input_dict['text']
    for i in sentences:
        temp = BertModel(i, "./classifi.ver1")
        if(temp==1): sentences.remove(i)
    input_dict['text'] = sentences
    return input_dict

if __name__ == '__main__':
    input_test = {'text' : ["샘플", "텍스트", "이런", "식으로"], 'image': r"0000000000000000"}
    input_test = DropAd(input_test)
    print(input_test['text'])
