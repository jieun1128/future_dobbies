from django.shortcuts import render 
from rest_framework.decorators import api_view 
from rest_framework.response import Response 
from rest_framework.views import APIView
from searchTest import insert_search_list, test_similarity
 

class HOME(APIView):
    def get(self, request):
        return render(request, 'home.html')


class searchTEXT(APIView):
    def get(self, request):

        return

