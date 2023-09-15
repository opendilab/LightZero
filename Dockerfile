# This Dockerfile describes the process of creating a Docker image that includes
# the necessary environment to run the LightZero library.

# The Docker image is based on Ubuntu 20.04, and it installs Python 3.8 and other
# necessary dependencies. It then clones the LightZero library from its GitHub
# repository and installs it in an editable mode.

# Before building the Docker image, create a new empty directory, move this Dockerfile into it,
# and navigate into this directory. This is to avoid sending unnecessary files to the Docker daemon
# during the build. Then you can then build the Docker image using the following command in your terminal:
# docker build -t ubuntu-py38-lz:latest -f ./Dockerfile .

# To run a container from the image in interactive mode with a Bash shell, you can use:
# docker run -dit --rm ubuntu-py38-lz:latest /bin/bash

# Once you're inside the container, you can run the example Python script with:
# python ./LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py

# Note: The working directory inside the Docker image is /opendilab, so you don't need
# to change your current directory before running the Python script.


# Start from Ubuntu 20.04
FROM ubuntu:20.04

# Set the working directory in the Docker image
WORKDIR /opendilab

# Install Python 3.8 and other dependencies
# We update the apt package list, install Python 3.8, pip, compilers and other necessary tools.
# After installing, we clean up the apt cache and remove unnecessary lists to save space.
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip gcc g++ swig git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for Python and pip
# This makes it easy to call python and pip from any location in the container.
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Update pip and setuptools to the latest version
# This step ensures that we have the latest tools for installing Python packages.
RUN python -m pip install --upgrade pip setuptools

# Clone the LightZero repository from GitHub
# This step downloads the latest version of LightZero to our Docker image.
RUN git clone https://github.com/opendilab/LightZero.git

# Install the LightZero package in editable mode
# The -e option allows us to edit the source code without needing to reinstall the package.
RUN pip install -e ./LightZero