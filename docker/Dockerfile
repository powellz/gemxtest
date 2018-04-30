ARG BASE_IMG
FROM $BASE_IMG

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-dev python3-pip 

RUN pip3 install --upgrade pip
RUN pip3 install keras scikit-learn pandas tensorflow

WORKDIR /opt 
