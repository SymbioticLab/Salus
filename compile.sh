#! /bin/bash
BUILD_TYPES=(Debug Release)

# Configure
for build_type in ${BUILD_TYPES[@]}; do
    binary_tree=build/$build_type
    [[ -d $binary_tree ]] || cmake -H. -B$binary_tree -DCMAKE_BUILD_TYPE=$biuld_type \
        -DTENSORFLOW_ROOT=$HOME/buildbed/tensorflow-rpcdev
done

# Build
for build_type in ${BUILD_TYPES[@]}; do
    cmake --build build/$build_type -- -j80
done
