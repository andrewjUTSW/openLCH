bootstrap: docker
from: nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

%post
# install Torch
apt-get update
apt-get install -y wget build-essential cmake git gnuplot5 libopenblas-dev torch7-nv 

# git proxy
git config --global url."https://".insteadOf git://

# install missing threads
rm -rf ~/.cache/luarocks
luarocks install threads
