export streamlit_app=True
app:

	streamlit run amadeusgpt/app.py --server.fileWatcherType none --server.maxUploadSize 1000

IMG_TAG := 0.2
IMG_NAME := amadeusgpt
DOCKERFILE := Dockerfile.sam2

BUILD_ARGS := \
        --build-arg uid=$(shell id -u) \
        --build-arg gid=$(shell id -g) \
        --build-arg user_name=$(shell id -un)
build:
	docker build $(BUILD_ARGS) \
		 -t $(IMG_NAME):$(IMG_TAG) -f $(DOCKERFILE) .

# [USER: ADJUST VOLUMES]
# path to the local project
HOST_SRC := /home/$(shell id -un)/AmadeusGPT
# path to the project in the container
DOCKER_SRC := /home/$(shell id -un)/AmadeusGPT
# DOCKER_DATA := /mnt
VOLUMES := \
	--volume $(HOST_SRC):$(DOCKER_SRC) 

CONTAINER_TAG :=v0.14
CONTAINER_NAME := amadeusgpt_$(CONTAINER_TAG)

run:
	docker run --shm-size=60G --gpus all -it --name $(CONTAINER_NAME) \
							$(VOLUMES) \
							$(IMG_NAME):$(IMG_TAG)
#  tail -f /dev/null

exec:
	docker exec -it $(CONTAINER_NAME) /bin/bash
