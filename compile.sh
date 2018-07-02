#! /bin/bash
set -e
BUILD_TYPES=(Debug Release TSan ASan OpTracing)

function configure() {
    local build_type=$1
    export CC=gcc-7
    export CXX=g++-7
    echo "Configure using build type $build_type"
    binary_tree=build/$build_type
    [[ -d $binary_tree ]] || cmake -H. -B$binary_tree -DCMAKE_BUILD_TYPE=$build_type \
        -DCMAKE_PREFIX_PATH=spack-packages \
        -DTENSORFLOW_ROOT=../tensorflow
}

if [[ $# > 0 ]]; then
    configure $1
    cmake --build build/$1 -- -j
    exit
fi

# Configure
for build_type in ${BUILD_TYPES[@]}; do
    configure $build_type
done

# Build
#for build_type in ${BUILD_TYPES[@]}; do
#    echo "Building $build_type"
#    cmake --build build/$build_type -- -j &
#done
#wait

export CLICOLOR_FORCE=1
ncores=$(($(nproc) / 4))
parallel --tag --lb -j 1 "cmake --build build/{} -- -j$ncores" ::: "${BUILD_TYPES[@]}"

