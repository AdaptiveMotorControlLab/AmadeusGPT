REPO=mmathislab/amadeus
CUDA_VERSION=11.4.0-cudnn8-runtime-ubuntu20.04
TAG=$(CUDA_VERSION)

IMAGE_NAME=$(REPO):$(TAG)
DOCKERFILE=Dockerfile
GPU=all
CONTAINER_NAME=amadeus
HOST_IP=5555

build_amadeus:
	@ echo Workdir: $(shell pwd)
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) \
                --build-arg CUDA_VERSION=$(CUDA_VERSION) ../../ 
run_gpu:
	docker run --rm -ti --gpus=$(GPU) --name $(CONTAINER_NAME)_$(GPU) -p $(HOST_IP):8501 $(IMAGE_NAME)

run_cpu:
	docker run --rm -ti --name $(CONTAINER_NAME) $(IMAGE_NAME) -p $(HOST_IP):8501 $(IMAGE_NAME)

debug:
	docker run --rm -ti -d --gpus=$(GPU) --name $(CONTAINER_NAME)_$(GPU) -p $(HOST_IP):8501 $(IMAGE_NAME) \
        && docker exec -ti $(CONTAINER_NAME)_$(GPU) bash
kill:
	docker rm -f $(CONTAINER_NAME)_$(GPU)

