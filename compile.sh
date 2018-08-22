#! /bin/bash
set -e
BUILD_TYPES=(Debug OpTracing Release TSan ASan)

function configure() {
    local build_type=$1
    which gcc-7 >/dev/null 2>&1 && export CC=gcc-7
    which g++-7 >/dev/null 2>&1 && export CXX=g++-7
    which ld.gold >/dev/null 2>&1 && export CXXFLAGS=-fuse-ld=gold
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

