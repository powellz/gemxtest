#!/bin/bash
set -e
uid=`id -u $USER`
sudo docker run -i -t -u $uid --name $uid-gemx --rm --net=host -v /dev/xdma:/dev/xdma -v /dev/xcldma:/dev/xcldma -v $PWD/..:/opt --privileged -it $uid/gemx
