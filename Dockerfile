FROM pytorch/pytorch:latest

ADD requirements.txt ./
RUN pip install -r requirements.txt
