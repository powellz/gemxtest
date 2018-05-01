#!/bin/bash

uid=`id -u $USER`

sudo docker build -t $uid/keras --build-arg BASE_IMG=ubuntu:16.04 -f Dockerfile .
