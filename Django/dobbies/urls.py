from django.conf.urls import url
from . import views

urlpatterns = [
    url('home', views.home, name='home'),
    url('found', views.found, name="found"),
    url('decod', views.decode, name="decode"),
]
