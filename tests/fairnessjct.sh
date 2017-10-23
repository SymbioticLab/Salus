#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TF_CPP_MIN_LOG_LEVEL=4

EXECUTOR=../build/Release/src/executor
BENCHMARKDIR=$HOME/buildbed/tf_benchmarks/scripts/tf_cnn_benchmarks

run_case() {
    local model=${2}
    local batch_size=${3}
    local outputfile=${4}
    local num_batches=${5:-20}

    local pid=0
    echo "Running $model of batch size $batch_size for $num_batches iterations"
    pushd $BENCHMARKDIR > /dev/null
    python tf_cnn_benchmarks.py --display_every=1 --local_parameter_device=cpu --num_gpus=1 \
                                --variable_update=parameter_server --nodistortions \
                                --num_batches=$num_batches \
                                --model=$model \
                                --batch_size=$batch_size \
                                > "$outputfile" &
    eval "$1=$!"
    popd > /dev/null
}

do_area() {
    local OUTPUTDIR=$1
    shift
    mkdir -p "$OUTPUTDIR"
    OUTPUTDIR=$(realpath $OUTPUTDIR)

    rm -f /tmp/err.output /tmp/alloc.output
    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    local pid=$!

    local workload_pids=()
    while (( "$#" )); do
        local wpid
        run_case wpid $1 $2 "$OUTPUTDIR/$1_$2_${3}iter.${#workload_pids[@]}.output" $3
        workload_pids+=("$wpid")
        shift 3
    done

    wait ${workload_pids[@]}

    kill $pid
    wait $pid
    mv /tmp/alloc.output $OUTPUTDIR/alloc.output
}

do_area ../scripts/logs/makespan_3of \
        overfeat 50 424 \
        overfeat 50 424 \
        overfeat 50 424

exit

do_area ../scripts/logs/area_3res \
        resnet50 50 265 \
        resnet50 50 265 \
        resnet50 50 265

exit

do_area ../scripts/logs/area_vgg \
        vgg11 25 20 \
        vgg11 50 20 \
        vgg11 100 20 \
        vgg19 25 20 \
        vgg19 50 20\
        vgg19 100 20
