from elasticsearch import Elasticsearch
from googleapiclient.discovery import build
import tensorflow_hub as hub
import numpy as np
import time
import os
import requests
from bs4 import BeautifulSoup

API_KEY = ''

ENGINE_ID = ''

embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
es = Elasticsearch(["elasticsearch"], PORT=9200, http_auth=("elastic", "123456"))
index_name="test_similarity"

def searchImage(text_url, image_url):
    search_result = []
    for i in image_url:
        store = True 
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=i, cx=ENGINE_ID).execute()
        for j in text_url:
            if res['items'][0]['link'] in j['url']:
                store = False 
        if store :
            search_temp = {}
            search_temp['url'] = res['items'][0]['link']
            search_temp['title'] = res['items'][0]['title']
            search_temp['snippet'] = res['items'][0]['snippet']
            search_result.append(search_temp)
    return search_result

# 검색 수행하기 
def search(search_text):
    search_result = []
    search_term = ''
    for i in search_text:
        if(len(i) < 32):
            search_term = i
        else:
            search_term = i[:32]
            break
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=search_term, cx=ENGINE_ID).execute()
        j=0
        for k in res['items']:
            search_temp = {}
            search_temp['url'] = k['link']
            search_temp['title'] = k['title']
            search_temp['snippet'] = k['snippet']
            search_result.append(search_temp)
            if(j+1 ==2): 
                break
            else:
                j+=1

    return search_result
    
def crawling(urls):
    i = 0
    for url in urls :
        webpage = requests.get(url['url'])
        title = url['title']
        snippet = url['snippet'].replace('\xa0',"")
        soup = BeautifulSoup(webpage.content, "html.parser")
        find = soup.find_all(['div', 'p'])
        description = ''
        for f in find : 
            text = f.get_text()
            text = text.replace('\n',"")
            text = text.replace('\xa0',"")
            description += text
        try:
            embeddings=embed([description])
            text_vector=np.array(embeddings[0]).tolist()
            doc={'idx':i+1,'text':description, 'title':title, 'snippet':snippet, 'url':url['url'], 'text-vector':text_vector}
        except:
            print('no data')
            break
        i+=1
        es.index(index=index_name, id=i, body=doc)
        
    es.indices.refresh(index=index_name)


    return 

# 유사도 검색하기
def test_similarity(search_text):
    query = ''
    for text in search_text:
        query += (text + ' ')
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
            "size": 7,
            "query":script_query,
            "_source": {"includes":["idx", "title", "snippet", "url"]}
        }
    )
    result=[]
    for hit in response["hits"]["hits"]:
        tmp={"id":hit["_source"]["idx"], "score":hit["_score"], "title":hit["_source"]["title"], "snippet":hit["_source"]["snippet"], "url":hit["_source"]["url"]}
        result.append(tmp)

    return result


def initialize_search_list():
    os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    # 인덱스 생성
    es.indices.create(index=index_name, body={
        "mappings":{
            "properties":{
                "idx":{
                    "type" :"integer"
                },
                "title":{
                    "type":"text"
                },
                "snippet":{
                    "type":"text"
                },
                "text":{
                    "type":"text"
                },
                "url":{
                    "type":"text"
                },
                "text-vector":{
                    "type": "dense_vector",
                    "dims": 512
                }
            }
        }
    })
    es.indices.refresh(index=index_name)




    


