from django.conf.urls import url
from . import views

urlpatterns = [
    url('home', views.HOME.as_view(), name='home'),
    url('search',  views.searchTEXT.as_view(), name='search')
]
