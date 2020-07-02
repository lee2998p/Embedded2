#usage:
# 'nvidia-docker build -f jetson-nano.dockerfile -t jetson .' to build docker image
# 'nvidia-docker run -ti jetson' to launch interactive shell in docker
#Ensure weight files are in jetson directory so they are added to image 

FROM nvcr.io/nvidia/l4t-pytorch:r32.4.2-pth1.5-py3

WORKDIR /cam2

ADD . /cam2/

RUN apt-get update -y && \
    apt-get install -y python3-opencv python3-pip python3-flask python3-scipy python3-matplotlib

RUN pip3 install -r requirements.txt


