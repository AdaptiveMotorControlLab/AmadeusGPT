# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# User arguments for container setup
ARG user_name
ARG uid
ARG gid

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean

# Create a non-root user
RUN mkdir -p /app /home/${user_name}
RUN groupadd -g ${gid} ${user_name} \
    && useradd -m -u ${uid} -g ${gid} ${user_name} \
    && chown -R ${uid}:${gid} /home/${user_name}

# Switch to the new user and set home directory as working directory
USER ${user_name}
ENV HOME=/home/${user_name}
WORKDIR ${HOME}

# Create a user conda environment to avoid permission issues
SHELL ["/bin/bash", "-c"]
RUN conda create -y --name amadeusGPT python=3.10
ENV PATH=/opt/conda/envs/amadeusGPT/bin:$PATH
RUN conda install -y -n amadeusGPT hdf5 pytables=3.8.0

# Install pip packages in the user environment
RUN pip install --no-cache-dir notebook amadeusgpt
RUN pip install -U --pre deeplabcut

# Initialize conda for the user's bash
RUN conda init bash
RUN echo 'conda activate amadeusGPT' >> ${HOME}/.bashrc

# Default command when container starts (activating the conda env)
CMD ["bash", "-l"]