#! /bin/bash
set -e
BUILD_TYPES=(Debug Release TSan ASan OpTracing)

# Configure
for build_type in ${BUILD_TYPES[@]}; do
    echo "Configure using build type $build_type"
    binary_tree=build/$build_type
    [[ -d $binary_tree ]] || cmake -H. -B$binary_tree -DCMAKE_BUILD_TYPE=$build_type \
        -DCMAKE_PREFIX_PATH=spack-packages \
        -DTENSORFLOW_ROOT=../tensorflow
done

# Build
#for build_type in ${BUILD_TYPES[@]}; do
#    echo "Building $build_type"
#    cmake --build build/$build_type -- -j &
#done
#wait

export CLICOLOR_FORCE=1
parallel --tag --lb -j 1 "cmake --build build/{} -- -j" ::: "${BUILD_TYPES[@]}"

