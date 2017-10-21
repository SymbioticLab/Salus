#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/src/executor
BENCHMARKDIR=$HOME/buildbed/benchmarks/scripts/tf_cnn_benchmarks

do_mem() {
    local OUTPUTDIR=$(realpath $1)
    mkdir -p $OUTPUTDIR

    echo "Running $2 of batch size $3"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/alloc.config &
    pid=$!
    pushd $BENCHMARKDIR > /dev/null
    python tf_cnn_benchmarks.py --display_every=1 --local_parameter_device=cpu --num_gpus=1 --variable_update=parameter_server --nodistortions \
                                --num_batches=20 \
                                --model=$2 --batch_size=$3 > $OUTPUTDIR/mem-iter.output
    #| tee $OUTPUTDIR/mem-iter.output
    popd
    kill $pid
    wait $pid
    mv /tmp/alloc.output $OUTPUTDIR/alloc.output
}

rm -f /tmp/err.output /tmp/alloc.output

do_mem ../scripts/logs/resnet50_25 resnet50 25
do_mem ../scripts/logs/resnet50_50 resnet50 50
do_mem ../scripts/logs/resnet50_75 resnet50 75

do_mem ../scripts/logs/resnet101_25 resnet101 25
do_mem ../scripts/logs/resnet101_50 resnet101 50
do_mem ../scripts/logs/resnet101_75 resnet101 75

do_mem ../scripts/logs/resnet152_25 resnet152 25
do_mem ../scripts/logs/resnet152_50 resnet152 50
do_mem ../scripts/logs/resnet152_75 resnet152 75

do_mem ../scripts/logs/googlenet_25 googlenet 25
do_mem ../scripts/logs/googlenet_50 googlenet 50
do_mem ../scripts/logs/googlenet_100 googlenet 100

do_mem ../scripts/logs/alexnet_25 alexnet 25
do_mem ../scripts/logs/alexnet_50 alexnet 50
do_mem ../scripts/logs/alexnet_100 alexnet 100

do_mem ../scripts/logs/overfeat_25 overfeat 25
do_mem ../scripts/logs/overfeat_50 overfeat 50
do_mem ../scripts/logs/overfeat_100 overfeat 100

do_mem ../scripts/logs/inception3_25 inception3 25
do_mem ../scripts/logs/inception3_50 inception3 50
do_mem ../scripts/logs/inception3_100 inception3 100

do_mem ../scripts/logs/inception4_25 inception4 25
do_mem ../scripts/logs/inception4_50 inception4 50
do_mem ../scripts/logs/inception4_75 inception4 75
