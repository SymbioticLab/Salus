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
    popd > /dev/null
}

do_prof() {
    local model=${1}
    local batch_size=${2}
    local OUTPUTDIR=${3:-"$LOGDIR/$1_$2"}

    mkdir -p $OUTPUTDIR
    local tempdir=$(mktemp -d --tmpdir do_prof.XXXXX)

    echo "Running $model of batch size $batch_size, temp dir: $tempdir"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 nvprof --export-profile "$tempdir/profile.sqlite" --events active_warps -f -- \
        $EXECUTOR --logconf ../build/disable.config &
    local pid=$!

    local wpid=''

    echo -n "    Running RPC: "
    run_case wpid $model $batch_size "$tempdir/rpc.output" "salus"
    wait $wpid
    mv "$tempdir/rpc.output" $OUTPUTDIR

    kill $pid
    wait $pid

    mv "$tempdir/profile.sqlite" $OUTPUTDIR
}

do_prof vgg11 25

do_prof vgg11 25
do_prof vgg11 50
do_prof vgg11 100

do_prof vgg16 25
do_prof vgg16 50
do_prof vgg16 100

do_prof vgg19 25
do_prof vgg19 50
do_prof vgg19 100

do_prof resnet50 25
do_prof resnet50 50
do_prof resnet50 75

do_prof resnet101 25
do_prof resnet101 50
do_prof resnet101 75

do_prof resnet152 25
do_prof resnet152 50
do_prof resnet152 75

do_prof googlenet 25
do_prof googlenet 50
do_prof googlenet 100

do_prof alexnet 25
do_prof alexnet 50
do_prof alexnet 100

do_prof overfeat 25
do_prof overfeat 50
do_prof overfeat 100

do_prof inception3 25
do_prof inception3 50
do_prof inception3 100

do_prof inception4 25
do_prof inception4 50
do_prof inception4 75
