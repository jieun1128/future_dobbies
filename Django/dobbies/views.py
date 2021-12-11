from logging import log
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
 

class HOME(APIView):
    def get(self, request):
        return render(request, 'home.html')


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

