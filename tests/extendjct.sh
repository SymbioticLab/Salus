#! /bin/sh
set -e

EXECUTOR=../build/Release/src/executor
BENCHMARKDIR=$HOME/buildbed/tf_benchmarks/scripts/tf_cnn_benchmarks
LOGDIR=templogs
workloads=(
vgg11 25 vgg11 50 vgg11 100
vgg16 25 vgg16 50 vgg16 100
vgg19 25 vgg19 50 vgg19 100
resnet50 25 resnet50 50 resnet50 75
resnet101 25 resnet101 50 resnet101 75
resnet152 25 resnet152 50 resnet152 75
googlenet 25 googlenet 50 googlenet 100
alexnet 25 alexnet 50 alexnet 100
overfeat 25 overfeat 50 overfeat 100
inception3 25 inception3 50 inception3 100
inception4 25 inception4 50 inception4 75
)

#=============================================================================
# Implementation detail, don't modify below this line
#=============================================================================

export CUDA_VISIBLE_DEVICES=0,1
export TF_CPP_MIN_LOG_LEVEL=4

die() {
    echo "$@"
    exit 1
}

run_case() {
    local model=${2}
    local batch_size=${3}
    local outputfile=${4}
    local executor=${5:-salus}
    local num_batches=${6:-20}

    echo "Running $model of batch size $batch_size for $num_batches iterations"
    pushd $BENCHMARKDIR > /dev/null
    stdbuf -o0 -e0 -- \
    python tf_cnn_benchmarks.py --display_every=1  --num_gpus=1 \
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

do_jct() {
    local model=${1}
    local batch_size=${2}
    local OUTPUTDIR=${3:-"$LOGDIR/$1_$2"}
    local batch_num=${BATCH_NUM:-20}

    mkdir -p $OUTPUTDIR

    echo "Running $model of batch size $batch_size"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    local pid=$!

    local wpid=''

    echo -n "    Warm up RPC: "
    run_case wpid $model $batch_size "/tmp/rpc.output" "salus" $batch_num
    wait $wpid
    mv /tmp/rpc.output $OUTPUTDIR
    echo

    echo -n "    Running RPC: "
    run_case wpid $model $batch_size "/tmp/rpc.output" "salus" $batch_num
    wait $wpid
    echo
    [[ -f /tmp/rpc.output ]] && mv /tmp/rpc.output $OUTPUTDIR

    echo -n "    Running GPU: "
    run_case wpid $model $batch_size "/tmp/gpu.output" "gpu" $batch_num
    wait $wpid
    echo
    [[ -f /tmp/gpu.output ]] && mv /tmp/gpu.output $OUTPUTDIR

    kill $pid
    wait $pid
}

do_jct_hint() {
    local model=$1
    local batch_size=$2
    local per_iter=$3
    local target_time=$4
    local tag=$5

    local OUTPUTDIR="$LOGDIR/${model}_${batch_size}_${tag}"
    mkdir -p "$OUTPUTDIR"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    local pid=$!

    # try until actual_time is within 10% of target_time
    echo "Finding suitable batch_num for $tag:"
    local base_gpu_jct=/tmp/gpu.output
    local actual_time=0
    local tried=0
    while (( $(bc <<< "define abs(x) {if (x<0) {return -x}; return x;}; abs($actual_time - $target_time) >= $target_time * 0.1") )); do
        # try 5 times or give up
        if (( tried > 5 )); then
            break
        fi

        local batch_num=$(bc <<< "$target_time / $per_iter")
        echo -n "        Trying batch_num=$batch_num: "

        run_case wpid $model $batch_size "$base_gpu_jct" "gpu" $batch_num
        wait $wpid
        echo
        [[ -f "$base_gpu_jct" ]] || die "$base_gpu_jct not found after running. Aborting"

        actual_time=$(awk -F'[^0-9.]*' '/^JCT/{print $2}' "$base_gpu_jct")
        # assume linear time distribution
        per_iter=$(bc <<< "scale=3; $actual_time / $batch_num")
        echo "            actual_time=$actual_time, per_iter=$per_iter"

        tried=$((tried + 1))
    done
    mv "$base_gpu_jct" "$OUTPUTDIR"

    # Use the batch_num to test RPC
    echo -n "    Warm up RPC: "
    local salus_jct=/tmp/rpc.output
    run_case wpid $model $batch_size "$salus_jct" "salus" $batch_num
    wait $wpid
    echo
    [[ -f "$salus_jct" ]] || die "$salus_jct not found after running. Aborting"

    echo -n "    Running RPC: "
    run_case wpid $model $batch_size "$salus_jct" "salus" $batch_num
    wait $wpid
    echo
    [[ -f "$salus_jct" ]] || die "$salus_jct not found after running. Aborting"
    mv "$salus_jct" "$OUTPUTDIR"

    kill $pid
    wait $pid
}

function main() {
    while [ ! -z "$1" ]  # while $1 is not empty
    do
        local network=$1
        local batch_size=$2
        printf "\n**** JCT: ${network}@${batch_size} *****\n\n"
        # Do base jct first
        do_jct $network $batch_size
        # Get per iter time
        local base_gpu_jct=$LOGDIR/${network}_${batch_size}/gpu.output
        if [[ ! -f "$base_gpu_jct" ]]; then
            echo "ERROR: $base_gpu_jct not found after running. Aborting"
            exit 1
        fi
        local per_iter=$(awk -F'[^0-9.]*' '/Average excluding/{print $2}' "$base_gpu_jct")

        do_jct_hint $network $batch_size \
            $per_iter \
            $((60 * 1)) \
            '1min'
        do_jct_hint $network $batch_size \
            $per_iter \
            $((60 * 5)) \
            '5min'
        do_jct_hint $network $batch_size \
            $per_iter \
            $((60 * 10)) \
            '10min'

        shift 2
    done
}

main "${workloads[@]}"
