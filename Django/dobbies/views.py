from django.db import reset_queries
from django.shortcuts import render 
from rest_framework.decorators import api_view 
from rest_framework.response import Response
from django.http import JsonResponse 
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .serealizer import SearchBodySerializer
from .searchTest import search, crawling, test_similarity, initialize_search_list, searchImage

# 필수 패키지 임포트
import base64
import io
import requests
# 경로 설정을 위한 패키지
import sys
sys.path.append("/home/jun/myWorks/GOD/others/future_dobbies/Image_Detection/modules") # 모튤 추가를 위한 시스템 패스 추가

# 추가된 경로에서 모듈 임포트
from gcpVision import get_vision_ocr, get_image_search
from imDetect import cropImg
 


def home(request):
    return render(request, 'home.html')


def found(request): # 안쓰는 url
    return render(request, 'found.html')

def decode(request):
    originImg = request.POST['hidden'] # 원본 이미지 파일 저장
    b64Img = originImg.split(';base64,')[1] # base64 인코딩 부분만 추출
    imgdecode = base64.b64decode(b64Img) # 바이트로 디코딩
    
    # 클라우드 비전을 활용한 검색 실시
    ocr = get_vision_ocr(imgdecode)
    img = cropImg(imgdecode)
    urls = get_image_search(img)
    
    # urls를 한번에 펼쳐서 준비
    tmp_urls = [u for url in urls for u in url ]

    # 검색 엔진으로 리퀘스트
    data = {'imageURL': tmp_urls, 'text': ocr}
    res = requests.post("http://0.0.0.0:8000/search", data=data)

    if res.status_code == 200: # 결과값이 있는 경우에만 디코딩 실시
        search_url = res.json
    else:
        search_url = "URL을 찾을 수 없습니다."

    return render(request, 'found.html', {'imgdc':originImg, 'texts': search_url})

class searchTEXT(APIView):
    @swagger_auto_schema(tags=["검색 수행"],
                         request_body=SearchBodySerializer,
                         responses={
                             200: "성공",
                             403: '인증에러',
                             400: '입력값 유효성 검증 실패',
                             500: '서버에러'
                         })
    def post(self, request):
        initialize_search_list()
        texts = request.data.get('text', None)
        search_urls = search(texts)
        
        try :
            image_urls = request.data.get('imageURL', None)
        except :
            image_urls = ''
        search_urls.extend(searchImage(search_urls, image_urls))

        crawling(search_urls)
        result = test_similarity(texts)

        return JsonResponse(result, safe=False)


