from django.shortcuts import render
from django.db import reset_queries
from rest_framework.decorators import api_view 
from rest_framework.response import Response 
from rest_framework.views import APIView
# from searchTest import insert_search_list, test_similarity
 


def home(request):
    return render(request, 'home.html')


def found(request):
    return render(request, 'found.html')

def decode(request):
    imgdecode = request.POST['hidden']
    return render(request, 'found.html', {'imgdc':imgdecode})

# class searchTEXT(APIView):
#     def get(self, request):

#         return

