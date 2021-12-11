FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED 1

RUN mkdir /django
WORKDIR /django

ADD requirements.txt /django/

RUN apt-get update
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade tensorflow-hub

ADD . /django/