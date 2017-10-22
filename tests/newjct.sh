#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/src/executor
BENCHMARKDIR=$HOME/buildbed/tf_benchmarks/scripts/tf_cnn_benchmarks
LOGDIR=templogs

run_case() {
    local model=${2}
    local batch_size=${3}
    local outputfile=${4}
    local executor=${5:-salus}
    local num_batches=${6:-20}

    echo "Running $model of batch size $batch_size for $num_batches iterations"
    pushd $BENCHMARKDIR > /dev/null
    python tf_cnn_benchmarks.py --display_every=1 --local_parameter_device=cpu --num_gpus=1 \
                                --variable_update=parameter_server --nodistortions \
                                --executor=$executor \
                                --num_batches=$num_batches \
                                --model=$model \
                                --batch_size=$batch_size \
                                > "$outputfile" &
    #| tee $OUTPUTDIR/mem-iter.output
    eval "$1=$!"
    popd
}

do_jct() {
    local model=${1}
    local batch_size=${2}
    local OUTPUTDIR=${3:-"$LOGDIR/$1_$2"}

    mkdir -p $OUTPUTDIR

    echo "Running $model of batch size $batch_size"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    local pid=$!

    local wpid=''

    echo -n "    Warm up RPC: "
    run_case wpid $model $batch_size "/tmp/rpc.output" "salus"
    wait $wpid
    mv /tmp/rpc.output $OUTPUTDIR

    echo -n "    Running RPC: "
    run_case wpid $model $batch_size "/tmp/rpc.output" "salus"
    wait $wpid
    mv /tmp/rpc.output $OUTPUTDIR

    echo -n "    Running GPU: "
    run_case wpid $model $batch_size "/tmp/gpu.output" "gpu"
    wait $wpid
    mv /tmp/gpu.output $OUTPUTDIR

    kill $pid
    wait $pid
}

do_jct vgg11 25
do_jct vgg11 50
do_jct vgg11 100

do_jct vgg16 25
do_jct vgg16 50
do_jct vgg16 100

do_jct vgg19 25
do_jct vgg19 50
do_jct vgg19 100

do_jct resnet50 25
do_jct resnet50 50
do_jct resnet50 75

do_jct resnet101 25
do_jct resnet101 50
do_jct resnet101 75

do_jct resnet152 25
do_jct resnet152 50
do_jct resnet152 75

do_jct googlenet 25
do_jct googlenet 50
do_jct googlenet 100

do_jct alexnet 25
do_jct alexnet 50
do_jct alexnet 100

do_jct overfeat 25
do_jct overfeat 50
do_jct overfeat 100

do_jct inception3 25
do_jct inception3 50
do_jct inception3 100

do_jct inception4 25
do_jct inception4 50
do_jct inception4 75