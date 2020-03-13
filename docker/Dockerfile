ARG SALUS_GCC_VERSION=8
#-----------------------------------
# Base builder image
#-----------------------------------
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 AS builder
# redeclare to make usable in this stage
ARG SALUS_GCC_VERSION

# Make sure we can always find CUDA
ENV CUDA_HOME=/usr/local/cuda

# Hold gcc 5 which otherwise will be upgraded in ubuntu-toolchain-r/test
# Install gcc-${SALUS_GCC_VERSION} and ld.gold and make them the default
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y --no-install-recommends software-properties-common gnupg-curl ca-certificates apt-transport-https curl \
    && apt-mark hold g++-5 \
    && apt-mark hold gcc-5 \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && (curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -) \
    && add-apt-repository -y 'https://apt.kitware.com/ubuntu/ xenial main' \
    && apt-get update \
    && apt-get install -y g++-${SALUS_GCC_VERSION} gcc-${SALUS_GCC_VERSION} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${SALUS_GCC_VERSION} 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${SALUS_GCC_VERSION} 20 \
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

#-----------------------------------
# Compilation steps used in CI
#-----------------------------------
FROM builder AS compile

COPY . /repo

ENV DESTDIR=/opt/salus-stage
RUN cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cmake --build build -- -j
RUN cmake --build build --target install -- DESTDIR=${DESTDIR}

# also copy over gcc runtime
RUN cp -a \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.* \
    /usr/lib/x86_64-linux-gnu/libgomp.so.* \
    /lib/x86_64-linux-gnu/libgcc_s.so.1 \
    /opt/salus-stage/usr/local/lib

# grab gosu for easy step-down from root
ENV GOSU_VERSION 1.11
RUN set -x \
    && apt-get -qq update && apt-get -qq install -y --no-install-recommends ca-certificates wget \
    && wget -O $DESTDIR/usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)" \
    && wget -O /gosu.asc "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture).asc" \
    && export GNUPGHOME="$(mktemp -d)" \
    && gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 \
    && gpg --batch --verify /gosu.asc $DESTDIR/usr/local/bin/gosu \
    && { command -v gpgconf > /dev/null && gpgconf --kill all || :; } \
    && rm -rf "$GNUPGHOME" /gosu.asc \
    && chmod +x $DESTDIR/usr/local/bin/gosu

#-----------------------------------
# Finally produce a smaller image
#-----------------------------------
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 AS prod
# redeclare to make usable in this stage
ARG SALUS_GCC_VERSION

COPY --from=compile /opt/salus-stage /

# needed to make sure libs we installed is preferred than system ones
ENV LD_LIBRARY_PATH=/usr/local/lib

# explicitly set user/group IDs
RUN set -eux; \
	groupadd -r salus --gid=999; \
	useradd -r -g salus --uid=999 --home-dir=/var/lib/salus --shell=/bin/bash salus; \
	mkdir -p /var/lib/salus; \
	chown -R salus:salus /var/lib/salus

COPY docker/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 5501

CMD ["salus-server"]
