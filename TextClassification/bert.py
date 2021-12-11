import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


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
    return np.argmax(logits) # return 0 or 1

def DropAd(input_dict):
    
    '''
    [Input]
    - input_dict : input data from OCR
    [Output]
    - input_dict : input data that non-ads are removed
    '''
    
    sentences = input_dict['text']
    for i in sentences:
        temp = BertModel(i, "./classifi.ver2")
        if(temp==1): sentences.remove(i)
    input_dict['text'] = sentences
    return input_dict    

if __name__ == '__main__':
    input_test = {'text' : ['NAVER 뉴스 TV연예 스포츠 뉴스스탠드 날씨 프리미엄', 'YMGYM -', '.', '뉴스홈', '속보', '정치 경제', '사회', '생활/문화', '세계', 'IT/과학', '오피니언', '포토', 'TV', '랭킹뉴스', '뉴스 검색', 'o', '12.09', '팩트체크 언론사 설정 언론사 뉴스 라이브러리', '머니S 랭킹 뉴스', '오후 4시~오후 5시까지 집계한 결과입니다.', 'Moneys', "위중증 857명 '역대 최다', 의료체계 결국 무너지나(종합)", '1 정은경 "12~17세 백신 미접종자, 접종', '자보다 감염률 2 …', '0 3시간전', '기사원문', '기사입력 2021.12.09. 오후 4:58', '정품', '스크랩', '본문듣기 · 설정', '20', '8', '요약', '가', 'R', '2 정은경 "12~17세 접종 이상반응, 인구', '10만명 당 2 …', '0 2시간전', "3 폭주하던 세종 아파트 실거래가 2.3억'", '뚝… 공인중개사 …', '0 2시간전', '4 비트코인, 오미크론 공포에도 6000만', '원대 보… 이더리움 …', '0 10시간전', '5 개가 물어서 치료비 달라고 하니… 견', "주 '돈 달라고 협박하 …", '0 3시간전', 'Moneys', '코로나19 확진자가 이틀연속 7000명대를 돌파한 9일 오전 서울 중구 서울역광장에 마련된 신종 코로나바이러스 감염증(코로나 19) 임시선별진', '료소에서 시민들이 검사를 받기 위해 줄을 서고 있다./사진=머니S 장동규 기자', '분야별 주요뉴스', '속 문 대통령, 김의철 KBS 사장 임명안 재가…34번째 野 패….', '신종 코로나바이러스 감염증(코로나19) 신규 확진자가 이틀 연속 7000명대를 기로했다. 위중증환자', ''], 'image': r"0000000000000000"}
    input_test = DropAd(input_test)
    print(input_test['text'])
