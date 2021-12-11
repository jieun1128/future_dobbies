from elasticsearch import Elasticsearch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import os

data = [
    "강자인 소니와 마이크로소프트(MS)가 최신형 비디오 게임기인 플레이스테이션5(PS5)와 X박스 시리즈X를 지난해 11월에 출시했고 현재로 1년을 맞",
    "[이뉴스투데이 김영민 기자] 차세대 콘솔 게임기 패권을 놓고 기대를 모았던 소니 플레이스테이션5(PS5)와 마이크로소프트 엑스박스 시리즈 X/S의",
    "지난해 11월 소니와 MS가 나란히 차세대 콘솔기기를 발표하면서 시장에서는 ... 소니 플레이스테이션과 마이크로소프트(MS) 엑스박스로 대표되는 콘솔",
    "[아이티데일리] 마이크로소프트(MS)와 소니가 새로운 비디오 게임기를 출시했다. MS는 500달러짜리 X박스 시리즈 X와 300달러짜리 시리즈 S를, 소니"
]

target = "비디오 게임기 시장 강자인 소니와 마이크로소프트(MS)가 최신형 비디오 게임기인 플레이스테이션5(PS5)와 X박스 시리즈X를 지난해 11월에 출시했고 현재로 1년을 맞았다."

# 유사도 검색하기
def test_similarity(target, embed):
    query = target
    embeddings=embed([query])
    query_vector=np.array(embeddings[0]).tolist()
    index_name="test_similarity"
    script_query={
        "script_score":{
            "query":{"match_all":{}},
            "script":{
                "source": "cosineSimilarity(params.query_vector, doc['text-vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    search_start=time.time()
    response=es.search(
        index=index_name,
        body={
            "size": 5,
            "query":script_query,
            "_source": {"includes":["idx", "text"]}
        }
    )
    result=[]
    search_time = time.time() - search_start
    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()
        tmp={"id":hit["_source"]["idx"], "score":hit["_score"]}
        result.append(tmp)

    print(result)
    return result

# 텍스트 임베딩 함수
def insert_search_list(embed):
    # data = list(cursor.fetchall())

    # 데이터 집어넣기
    for i in range(len(data)):
        try:
            description=data[i].replace("\n"," ").replace("'",'').replace('"','').strip()
        except:
            description=""
        try:
            embeddings=embed([description])
            text_vector=np.array(embeddings[0]).tolist()
            doc={'idx':i+1,'text':description, 'text-vector':text_vector}
        except:
            print('no data')
            print('마지막 인덱스', i)
            break
        es.index(index=index_name, id=i, body=doc)
        
    es.indices.refresh(index=index_name)

def initialize_search_list():
    os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"
    embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
    #insert_search_list(embed)

    index_name="test_similarity"
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    # 인덱스 생성
    es.indices.create(index=index_name, body={
        "mappings":{
            "properties":{
                "idx":{
                    "type" :"integer"
                },
                "text":{
                    "type":"text"
                },
                "text-vector":{
                    "type": "dense_vector",
                    "dims": 512
                }
            }
        }
    })
    #test_similarity(target,embed)


if __name__ == '__main__':
    es = Elasticsearch(["elasticsearch"], PORT=9200, http_auth=("elastic", "123456"))

    initialize_search_list()



    


