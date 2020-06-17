# use python 3.7 as base image
FROM python:3.7


# install dependencies
RUN pip install tensorflow==2.2.0
RUN pip freeze > requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/mugundh/Pictures/Dockerfile

COPY trail /trail

COPY my_model /
COPY imagelabels.mat /
COPY README.md /
COPY kerascheck.py /


#Run  when container is launced
CMD ./kerascheck.py
