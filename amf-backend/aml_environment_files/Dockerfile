# Start from a base image with libvips
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# Switch to root to install packages
USER root:root

# Install libvips and other needed dependencies
RUN apt-get update && \
    apt-get install -y libvips libvips-dev gcc g++ pkg-config wget  && \
    ldconfig && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set the working directory
WORKDIR /app

# Copy the requirements_aml.yml file into the image, create conda environment and add to path
COPY requirements_aml.yml .
RUN conda env create -f requirements_aml.yml && \
    conda clean -afy

#Installing azureml-mlflow
RUN pip install azureml-mlflow azureml-core azureml-mlflow

# Set the command to execute
CMD ["exec bash"]