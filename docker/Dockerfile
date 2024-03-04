# image with depracation notice
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 # deprecated
# image with problem with tf and cuda drivers
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install numpy pandas pyarrow matplotlib xgboost scikit-learn nflows lightgbm seaborn iminuit
# Install TensorFlow and PyTorch
RUN pip3 install keras
RUN pip3 install tensorflow[and-cuda] 
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CUDNN
# RUN pip3 install nvidia-pyindex
# RUN pip3 install nvidia-cudnn

# Set the default command to Python3
CMD ["python3"]


# To build image from docker file
# docker build -t fair_universe --platform linux/amd64 .

# To run this image
# docker run -it fair_universe

# get container id from running image
# then exec with the following command
# docker exec -it container_id /bin/bash

# copy bundle to docker container
# docker cp /your/local/directory your_container_name:/app


# tag your image 
# docker tag fair_universe ihsaanullah/fair_universe:latest

# push image
# docker push ihsaanullah/fair_universe:latest

