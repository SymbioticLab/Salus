#-----------------------------------
# Base builder image
#-----------------------------------
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 AS builder

# Make sure we can always find CUDA
ENV CUDA_HOME=/usr/local/cuda

# Hold gcc 5 which otherwise will be upgraded in ubuntu-toolchain-r/test
# Install gcc-7 and ld.gold and make them the default
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y --no-install-recommends software-properties-common gnupg-curl ca-certificates apt-transport-https curl \
    && apt-mark hold g++-5 \
    && apt-mark hold gcc-5 \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && (curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -) \
    && add-apt-repository -y 'https://apt.kitware.com/ubuntu/ xenial main' \
    && apt-get update \
    && apt-get install -y g++-9 gcc-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 \
    && update-alternatives --set cc /usr/bin/gcc \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 \
    && update-alternatives --set c++ /usr/bin/g++ \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 20 \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.bfd 10 \
    && apt-get install -y cmake git rsync python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache -U pip

# Python
RUN pip3 install --no-cache -U conan invoke \
    && conan profile new default --detect \
    && conan profile update settings.compiler.libcxx=libstdc++11 default

WORKDIR /repo
CMD ["/bin/bash"]

#-----------------------------------
# Additional build server for CLion
#-----------------------------------

FROM builder AS clion

# Install ssh server and other development tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y openssh-server gdb

# Config the sshd server, the root password is 'root'.
# For development use ONLY, NEVER expose this to the Internet!!!
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && mkdir /root/.ssh

# Make sure environment variables are set for ssh sessions
RUN /bin/echo -e "\
PATH=${PATH}\n\
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\n\
" >> /etc/environment

VOLUME /root

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
