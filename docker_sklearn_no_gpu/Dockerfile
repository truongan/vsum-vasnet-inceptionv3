FROM tensorflow/tensorflow:2.7.1

COPY tvsum_splits_1_0.6187176441788906.tar.pth.tar /working/model.pth.tar
# COPY *.py /working/

RUN pip install ortools
RUN pip install opencv-python
RUN pip install torch
RUN pip install tabulate
RUN pip install pandas
RUN pip install sklearn


RUN apt update && apt install -y ffmpeg parallel 
RUN apt update && apt install -y openvpn