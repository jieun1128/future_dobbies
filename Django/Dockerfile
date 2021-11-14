FROM python:3.7

ENV PYTHONUNBUFFERED 1

RUN mkdir /django
WORKDIR /django

ADD requirements.txt /django/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade tensorflow-hub

ADD . /django/