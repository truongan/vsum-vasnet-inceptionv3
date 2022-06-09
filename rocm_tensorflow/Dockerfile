FROM rocm/tensorflow:rocm5.1.3-tf2.8-horovod-dev


#COPY tvsum_splits_1_0.6187176441788906.tar.pth.tar /working/model.pth.tar
# COPY *.py /working/

RUN pip install ortools
RUN pip install opencv-python
RUN pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.1.1/
RUN pip install tabulate
RUN pip install pandas
RUN pip install sklearn
RUN pip install psutil


RUN apt update && apt install -y ffmpeg parallel 

ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# RUN apt update && apt install -y openvpn