ARG CUDA_VERSION=11.4.0-cudnn8-runtime-ubuntu20.04
FROM mmathislab/amadeus:${CUDA_VERSION}

WORKDIR /app

RUN apt-get update -yy \ 
    && apt-get install -yy --no-install-recommends python3 python3-pip python3-dev ffmpeg libsm6 libxext6 git\
    && apt-get install -yy --no-install-recommends build-essential make cmake gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=${PATH}:/root/.local/bin

#Note: if changed, adjust path in entrypoint (didn't work via arg)
ARG SRC_CODE=amadeusGPT-streamlit
COPY ${SRC_CODE} ${SRC_CODE}

ARG REQ_FILE=${SRC_CODE}/requirements.txt
COPY ${REQ_FILE} ${REQ_FILE}

RUN pip3 install --upgrade --user -r ${REQ_FILE}

ARG DEEPLABCUT_VERSION=2.3.4
ARG CEBRA_VERSION=0.2.0

RUN pip3 install --upgrade \
		deeplabcut==${DEEPLABCUT_VERSION} \
		cebra==${CEBRA_VERSION} \
		torch==1.12 \
		tensorflow==2.5 \
		numpy==1.23.1 \
		scipy==1.9.1 \
		streamlit==1.21.0 \
		streamlit[extras]

#get clone
ARG SEGMENT_ANYTHING=segment-anything
RUN pip3 install ${SEGMENT_ANYTHING}

ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEEPLABCUT_VERSION=${DEEPLABCUT_VERSION}
ENV CEBRA_VERSION=${CEBRA_VERSION}
ENV streamlit_app=True

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR amadeusGPT-streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.fileWatcherType",  "none", "--server.port=8501", "--server.address=0.0.0.0"]

