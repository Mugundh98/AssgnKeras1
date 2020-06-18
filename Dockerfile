# use python 3.7 as base image
FROM ubuntu:latest
FROM python:latest


# install dependencies
RUN pip install tensorflow==2.2.0
COPY requirements.txt /
RUN pip install -r requirements.txt

COPY trail /trail

COPY my_model /
COPY imagelabels.mat /
COPY README.md /
COPY kerascheck.py /
RUN chmod u+x kerascheck.py

#Run  when container is launced
CMD ./kerascheck.py
