# use python 3.7 as base image

FROM ubuntu:16.04
FROM python:3.7


# install dependencies
COPY requirements.txt /

RUN pip install -r requirements.txt

COPY trail / trail /

COPY my_model /
COPY imagelabels.mat /
COPY README.md /
COPY kerascheck.py /

RUN chmod u+x kerascheck.py

#Run  when container is launced
CMD ./kerascheck.py
