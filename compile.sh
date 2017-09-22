#! /bin/bash
mkdir -p build && cd build

BUILD_TYPES=(Debug Release)

# Configure
for build_type in ${BUILD_TYPES[@]}; do
    mkdir -p $build_type
    pushd $build_type >/dev/null

    cmake -DTENSORFLOW_ROOT=$HOME/buildbed/tensorflow-rpcdev -DCMAKE_BUILD_TYPE=$build_type ../..

    popd >/dev/null
done

# Build
for build_type in ${BUILD_TYPES[@]}; do
    pushd $build_type >/dev/null
    make -j80
    popd >/dev/null
done
