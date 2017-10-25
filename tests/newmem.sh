#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/src/executor
BENCHMARKDIR=$HOME/buildbed/tf_benchmarks/scripts/tf_cnn_benchmarks
LOGDIR=templogs

do_mem() {
    local OUTPUTDIR=$1
    mkdir -p $OUTPUTDIR

    local tempdir=$(mktemp -d --tmpdir newmem.XXXXXX)

    echo "Running $2 of batch size $3"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/alloc.config &
    local pid=$!
    pushd $BENCHMARKDIR > /dev/null
    stdbuf -o0 -e0 -- \
    python tf_cnn_benchmarks.py --display_every=1 --local_parameter_device=cpu --num_gpus=1 --variable_update=parameter_server --nodistortions \
                                --num_batches=20 \
                                --model=$2 --batch_size=$3 > $tempdir/mem-iter.output
    #| tee $OUTPUTDIR/mem-iter.output
    popd > /dev/null
    kill $pid
    wait $pid
    mv /tmp/alloc.output $OUTPUTDIR/alloc.output
    mv $tempdir/* $OUTPUTDIR
    rmdir $tempdir
}

rm -f /tmp/err.output /tmp/alloc.output

do_mem $LOGDIR/vgg11_25 vgg11 25
do_mem $LOGDIR/vgg11_50 vgg11 50
do_mem $LOGDIR/vgg11_100 vgg11 100

do_mem $LOGDIR/vgg16_25 vgg16 25
do_mem $LOGDIR/vgg16_50 vgg16 50
do_mem $LOGDIR/vgg16_100 vgg16 100

do_mem $LOGDIR/vgg19_25 vgg19 25
do_mem $LOGDIR/vgg19_50 vgg19 50
do_mem $LOGDIR/vgg19_100 vgg19 100

do_mem $LOGDIR/resnet50_25 resnet50 25
do_mem $LOGDIR/resnet50_50 resnet50 50
do_mem $LOGDIR/resnet50_75 resnet50 75

do_mem $LOGDIR/resnet101_25 resnet101 25
do_mem $LOGDIR/resnet101_50 resnet101 50
do_mem $LOGDIR/resnet101_75 resnet101 75

do_mem $LOGDIR/resnet152_25 resnet152 25
do_mem $LOGDIR/resnet152_50 resnet152 50
do_mem $LOGDIR/resnet152_75 resnet152 75

do_mem $LOGDIR/googlenet_25 googlenet 25
do_mem $LOGDIR/googlenet_50 googlenet 50
do_mem $LOGDIR/googlenet_100 googlenet 100

do_mem $LOGDIR/alexnet_25 alexnet 25
do_mem $LOGDIR/alexnet_50 alexnet 50
do_mem $LOGDIR/alexnet_100 alexnet 100

do_mem $LOGDIR/overfeat_25 overfeat 25
do_mem $LOGDIR/overfeat_50 overfeat 50
do_mem $LOGDIR/overfeat_100 overfeat 100

do_mem $LOGDIR/inception3_25 inception3 25
do_mem $LOGDIR/inception3_50 inception3 50
do_mem $LOGDIR/inception3_100 inception3 100

do_mem $LOGDIR/inception4_25 inception4 25
do_mem $LOGDIR/inception4_50 inception4 50
do_mem $LOGDIR/inception4_75 inception4 75
