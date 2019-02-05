# Defining environment
ARG APP_ENV=dev
ARG BUILD_TYPE=Debug

FROM scratch as spackbc

#-----------------------------------
# Base image
#-----------------------------------
FROM registry.gitlab.com/salus/tensorflow-salus AS base-prod

# make gcc-7 and ld.gold the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 20 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 \
    && update-alternatives --set cc /usr/bin/gcc \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 \
    && update-alternatives --set c++ /usr/bin/g++ \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 20 \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.bfd 10

ENV SALUS_DEPS_DIR=/opt/salus-deps

#-----------------------------------
# Additional build server for CLion in base
#-----------------------------------

FROM base-prod AS base-dev

# Install ssh server and other development tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y openssh-server

RUN spack install gdb \
    && spack view -d false -v add $SPACK_PACKAGES gdb \
    && spack-pin gdb

# Config the sshd server, the root password is 'root'.
# For development use ONLY, NEVER expose this to the Internet!!!
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && mkdir /root/.ssh

#-----------------------------------
# Dependencies
#-----------------------------------
FROM base-${APP_ENV} AS deps

# Update spack
RUN cd $SPACK_HOME && git pull

RUN spack install boost@1.66.0
RUN spack install cppzmq@4.3.0 \
                  zeromq@4.2.5 \
                  nlohmann-json@3.1.2 \
                  protobuf@3.4.1 \
                  gperftools@2.7

RUN spack view -v -d false hard "$SALUS_DEPS_DIR" boost@1.66.0 \
                                                  cppzmq@4.3.0 \
                                                  zeromq@4.2.5 \
                                                  nlohmann-json@3.1.2 \
                                                  protobuf@3.4.1 \
                                                  gperftools@2.7

ENV CMAKE_PREFIX_PATH=$SALUS_DEPS_DIR

#-----------------------------------
# Add in source code
#-----------------------------------
FROM deps AS sources

COPY . salus

#-----------------------------------
# Development image
#-----------------------------------
FROM sources AS dev

# Make sure environment variables are set for ssh sessions
RUN /bin/echo -e "\
PATH=${PATH}\n\
SALUS_WORK_ROOT=${SALUS_WORK_ROOT}\n\
SPACK_HOME=${SPACK_HOME}\n\
SPACK_PACKAGES=${SPACK_PACKAGES}\n\
CUDA_HOME=${CUDA_HOME}\n\
TensorFlow_DIR=${TensorFlow_DIR}\n\
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\n\
" >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

#-----------------------------------
# Production image
#-----------------------------------

FROM sources as compile

WORKDIR /salus/salus

ENV Salus_DIR=${SALUS_WORK_ROOT}/salus

RUN cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/usr -DSALUS_DEPS_PATH=$SALUS_DEPS_DIR

RUN cmake --build build -- -j

RUN cmake --build build --target install -- DESTDIR=/opt/salus

# build a smaller image
FROM deps as prod

COPY --from=compile /opt/salus /usr

EXPOSE 5501

CMD ["salus-server"]

#-----------------------------------
# Final image as a switch
#-----------------------------------
FROM ${APP_ENV} as final
