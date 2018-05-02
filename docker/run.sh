#!/bin/bash
set -e
user=root
sudo docker run -i -t -u $user --name $user-keras --rm --net=host -v /dev/xdma:/dev/xdma -v /dev/xcldma:/dev/xcldma -v $PWD/..:/opt --privileged -it $user/keras

