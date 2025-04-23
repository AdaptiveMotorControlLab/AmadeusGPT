FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ARG user_name
ARG uid
ARG gid
ENV HOME=/root 

WORKDIR ${HOME}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    vim zsh tmux wget curl htop jupyter python3 python3-pip libgl1-mesa-glx git sudo ssh libglib2.0-0 \
    tree \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean \
    && apt-get update 

# Create a non-root user
RUN mkdir /app /logs /data
RUN groupadd -g ${gid} ${user_name} \
    && useradd -m -u ${uid} -g ${gid} ${user_name} \
    && chown -R ${uid}:${gid} /home
# 
# Switch to the new user and set home directory as working directory
USER ${user_name}
ENV HOME=/home/${user_name}
WORKDIR ${HOME}

# Install Oh My Zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && sed -i 's/^plugins=(.*)$/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

RUN echo "export PATH=$PATH:/home/${user_name}/.local/bin" >> /home/${user_name}/.zshrc

# install pip packages
COPY ./conda/amadeusGPT.yml ${HOME}/amadeusGPT.yml
SHELL ["/bin/bash", "-lc"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f ${HOME}/amadeusGPT.yml && \
    conda activate amadeusgpt && \
    pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir gpustat nvitop pytest PySide6==6.3.1 streamlit networkx isort black git+https://github.com/DeepLabCut/DeepLabCut.git

# USER ${user_name}
USER root
# ENV HOME=/root
WORKDIR /app

CMD ["zsh"]
SHELL ["/bin/zsh", "-c"]