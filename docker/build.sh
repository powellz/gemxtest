#!/bin/bash

uid=`id -u $USER`
gid=`id -g $USER`

sudo docker build -t $USER/keras --build-arg BASE_IMG=ubuntu:16.04 -f Dockerfile .
