#! /bin/bash

#-------------------------------------------------------
# Settings
#-------------------------------------------------------
OUTPUT_DIR=$HOME/vgg16-rpc-gpu
CASE='test_tf.test_vgg'
RPC_ONLY_METHOD='TestVgg16.test_rpc_only'
BOTH_METHOD='TestVgg16.test_fake_data'

PYTHON=$HOME/.local/venvs/tfbuild/bin/python

#-------------------------------------------------------
# Implementations
#-------------------------------------------------------
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXEC_DIR=$DIR/..

function run_client() {
    pushd $EXEC_DIR/tests > /dev/null
    env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m $CASE $1 | tee "$2"
    popd > /dev/null
}

function run_server() {
    env CUDA_VISIBLE_DEVICES=2,3 $EXEC_DIR/build/src/executor -vvv > "$1" 2>&1 &
}

function profile_server() {
    env TF_CPP_MIN_LOG_LEVEL=4 CUDA_VISIBLE_DEVICES=2,3 nvprof --export-profile "$1" --events active_warps -- $EXEC_DIR/build/src/executor &
}

rm -rf "$OUTPUT_DIR" && mkdir -p "$OUTPUT_DIR"
cd $EXEC_DIR

# Collect memory usage

run_server "$OUTPUT_DIR/exec.output"
run_client $RPC_ONLY_METHOD "$OUTPUT_DIR/mem-iter.log"
kill $(jobs -p)

# Collect JCT
for i in $(seq 3); do
    run_server /dev/null
    run_client $BOTH_METHOD "$OUTPUT_DIR/$i.speed"
    kill $(jobs -p)
done

# Collect Compute
profile_server "$OUTPUT_DIR/profile.sqlite"

run_client $RPC_ONLY_METHOD "$OUTPUT_DIR/com-iter.log"

kill $(jobs -p)
