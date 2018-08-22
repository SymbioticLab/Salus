FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# Set the working directory to /salus
WORKDIR /salus

# Copy the current directory contents into the container at /app
# COPY . /salus

# Setup base system repositories
# gcc/g++ 5 need to be hold otherwise will be upgraded to 5.5.0 in ubuntu-toolchain-r/test
RUN apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y software-properties-common \
    && apt-mark hold g++-5 \
    && apt-mark hold gcc-5 \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && add-apt-repository -y ppa:webupd8team/java \
    && apt-get update

# Install base system packages
RUN apt-get install -y g++-7 gcc-7 \
    && (echo 'oracle-java8-installer shared/accepted-oracle-license-v1-1 select true' | /usr/bin/debconf-set-selections) \
    && apt-get install -y oracle-java8-installer oracle-java8-set-default \
    && apt-get install -y python python-dev python-pip python-wheel \
    && apt-get install -y git curl openssl libssl-dev \
    && pip install -U pip

# Re-enable interactive debconf
RUN (echo 'debconf debconf/frontend select Dialog' | debconf-set-selections)

# Install spack
ENV PATH="/salus/spack/bin:/salus/packages/bin:${PATH}"
COPY spack/packages.yaml /root/.spack/packages.yaml
RUN git clone https://github.com/spack/spack.git \
    && spack compilers \
    && spack install cmake \
    && spack view -d false -v add packages cmake

# Install python dependencies
RUN pip install --trusted-host pypi.python.org invoke

# Run app.py when the container launches
CMD ["bash"]
