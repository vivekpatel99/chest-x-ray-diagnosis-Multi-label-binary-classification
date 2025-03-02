# This assumes the container is running on a system with a CUDA GPU
#ähttps://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-01.html
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set DEBIAN_FRONTEND temporarily for build-time only
ARG DEBIAN_FRONTEND=noninteractive

# ARG USER_UID
# ARG USER_GID
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set environment variable to enable dynamic GPU memory allocation
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_VISIBLE_DEVICES=0
ENV HOME=/home/$USERNAME

# Remove the existing group and user if they already exist
# RUN if getent group ${USER_GID} > /dev/null; then delgroup $(getent group ${USER_GID} | cut -d: -f1); fi
# RUN if getent passwd ${USER_UID} > /dev/null; then deluser $(getent passwd ${USER_UID} | cut -d: -f1); fi

# Add the group and user with the specified GID and UID
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m -d $HOME $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME   

# Create the user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m -d $HOME $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME   

RUN apt-get update -y && apt-get upgrade -y  \
    && apt-get install -y sudo wget curl unzip git ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV setup and sync
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER $USERNAME
WORKDIR $HOME
# RUN sudo chown -R $USERNAME:$USERNAME $HOME
RUN pip install tensorflow==2.18.0 hydra-core ipykernel ipywidgets tqdm seaborn opendatasets mlflow

# ENV VIRTUAL_ENV="$HOME/.venv" 
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY pyproject.toml .
# RUN uv sync --active 


