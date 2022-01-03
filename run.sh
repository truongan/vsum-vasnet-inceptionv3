#!/bin/bash

# for i in /dev/nvidia*; do  echo -n " --device $i" ; done
# docker run -it --rm --gpus all --device /dev/nvidia0  --device /dev/nvidia-modeset --device /dev/nvidiactl -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
