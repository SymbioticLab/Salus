#!/bin/bash
# Created Date: Friday, August 23rd 2019, 5:04:12 pm
# Author: Yuchen Zhong
# Email: yczhong@hku.hk

export CUDA_VISIBLE_DEVICES=0

rm -rf /tmp/server.output

dir=`echo $0 | cut -d / -f2 | cut -d . -f1`
log_dir=templogs/$dir
mkdir -p $log_dir

salus_dir="/salus/salus/build/Debug/bin"
salus_bin="salus-server"

benchmark_dir="../../tf_benchmarks/scripts/tf_cnn_benchmarks"
exe="tf_cnn_benchmarks.py"

# only needs to change workloads
WORKLOADS=(
    "resnet50_25_250"
    "resnet50eval_1_2000"
    "resnet50eval_1_2000"
    "resnet50eval_1_2000"
)


COMMANDS=()
for (( i=0;i<${#WORKLOADS[@]};i++ )); do
    workload=${WORKLOADS[$i]}
    model=`echo ${workload} | cut -d _ -f1`
    batch_size=`echo ${workload} | cut -d _ -f2`
    num_batches=`echo ${workload} | cut -d _ -f3`
    cmd="python ${exe} \
        --model=${model} \
        --batch_size=${batch_size} \
        --num_batches=${num_batches} \
        --variable_update=parameter_server \
        --distortions=false \
        --num_gpus=1 \
        --display_every=1 "
    
    if [[ $workload =~ "eval" ]]; then
        cmd=${cmd}"--eval \
                  --eval_block=true \
                  --eval_interval_random_factor=5 \
                  --eval_interval_secs=1 \
                  --train_dir=models/${model} \
                  --model=${model} "
    fi

    # logs
    cmd=${cmd}"> ${log_dir}${model}.output 2>&1 &"
    # echo $cmd
    COMMANDS[$i]="${cmd}"
done

echo ${#COMMANDS[@]} "jobs are loaded"

cd $benchmark_dir

POLICIES=(
    "pack"
    "fair"
    "preempt"
    "mix"
)

for policy in ${POLICIES[@]}; do
    mkdir -p $log_dir/$policy
    printf "\n***** SALUS START *****\n"
    eval "${salus_dir}/${salus_bin} -s ${policy} -v 9 -c /salus/salus/scripts/logconf/disable.config &"

    for cmd in ${COMMANDS[@]}; do
        printf "\n***** Running: ${cmd} *****\n"
        eval $cmd
    done

    mv /tmp/server.output $log_dir/$policy
done