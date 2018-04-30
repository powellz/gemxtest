#!/bin/bash
set -e

sudo docker run -i -t -u $USER --name $USER-keras --rm --net=host -v /dev/xdma:/dev/xdma -v /dev/xcldma:/dev/xcldma -v $PWD/..:/opt --privileged -it $USER/keras
