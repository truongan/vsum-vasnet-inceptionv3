#!/bin/bash

# for i in /dev/nvidia*; do  echo -n " --device $i" ; done
# docker run -it --rm --gpus all --device /dev/nvidia0  --device /dev/nvidia-modeset --device /dev/nvidiactl -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
# docker run -it --rm  -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
docker run -it   -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash /current/run_inside.sh $@