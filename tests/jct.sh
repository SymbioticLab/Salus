#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/src/executor

do_jct() {
    local OUTPUTDIR=$1
    mkdir -p $OUTPUTDIR

    echo "Running $2.$3"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    pid=$!

    echo "    warm up: $2.$3.$4"
    python -m $2 $3.$4 > $OUTPUTDIR/rpc.output
    echo "    running: $2.$3.$4"
    python -m $2 $3.$4 > $OUTPUTDIR/rpc.output
    echo "    running: $2.$3.$5"
    python -m $2 $3.$5 > $OUTPUTDIR/gpu.output

    kill $pid
    wait $pid
}

do_jct ../scripts/logs/ptbT test_tf.test_seq TestSeqPtb test_rpc_0_tiny test_gpu_0_tiny
do_jct ../scripts/logs/ptbS test_tf.test_seq TestSeqPtb test_rpc_1_small test_gpu_1_small
do_jct ../scripts/logs/ptbM test_tf.test_seq TestSeqPtb test_rpc_2_medium test_gpu_2_medium
do_jct ../scripts/logs/ptbL test_tf.test_seq TestSeqPtb test_rpc_3_large test_gpu_3_large

exit

do_jct ../scripts/logs/conv25 test_tf.test_mnist_tf TestMnistConv test_rpc_0 test_gpu_0
do_jct ../scripts/logs/conv50 test_tf.test_mnist_tf TestMnistConv test_rpc_1 test_gpu_1
do_jct ../scripts/logs/conv100 test_tf.test_mnist_tf TestMnistConv test_rpc_2 test_gpu_2

do_jct ../scripts/logs/mnist25 test_tf.test_mnist_tf TestMnistLarge test_rpc_0 test_gpu_0
do_jct ../scripts/logs/mnist50 test_tf.test_mnist_tf TestMnistLarge test_rpc_1 test_gpu_1
do_jct ../scripts/logs/mnist100 test_tf.test_mnist_tf TestMnistLarge test_rpc_2 test_gpu_2

do_jct ../scripts/logs/vgg25 test_tf.test_vgg TestVgg16 test_rpc_only_0 test_gpu_0
do_jct ../scripts/logs/vgg50 test_tf.test_vgg TestVgg16 test_rpc_only_1 test_gpu_1
do_jct ../scripts/logs/vgg100 test_tf.test_vgg TestVgg16 test_rpc_only_2 test_gpu_2

do_jct ../scripts/logs/res25 test_tf.test_resnet TestResNetFakeData test_rpc_0 test_gpu_0
do_jct ../scripts/logs/res50 test_tf.test_resnet TestResNetFakeData test_rpc_1 test_gpu_1
do_jct ../scripts/logs/res75 test_tf.test_resnet TestResNetFakeData test_rpc_2 test_gpu_2

do_jct ../scripts/logs/ptbT test_tf.test_seq TestSeqPtb test_rpc_0_tiny test_gpu_0_tiny
do_jct ../scripts/logs/ptbS test_tf.test_seq TestSeqPtb test_rpc_1_small test_gpu_1_small
do_jct ../scripts/logs/ptbM test_tf.test_seq TestSeqPtb test_rpc_2_medium test_gpu_2_medium
do_jct ../scripts/logs/ptbL test_tf.test_seq TestSeqPtb test_rpc_3_large test_gpu_3_large
