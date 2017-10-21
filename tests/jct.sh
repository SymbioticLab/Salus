#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/src/executor
LOGDIR=templogs

do_jct() {
    local OUTPUTDIR=$1
    mkdir -p $OUTPUTDIR

    echo "Running $2.$3"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/disable.config &
    pid=$!

    echo "    warm up: $2.$3.$4"
    python -m $2 $3.$4 > /tmp/rpc.output
    mv /tmp/rpc.output $OUTPUTDIR
    echo "    running: $2.$3.$4"
    python -m $2 $3.$4 > /tmp/rpc.output
    mv /tmp/rpc.output $OUTPUTDIR
    echo "    running: $2.$3.$5"
    python -m $2 $3.$5 > /tmp/gpu.output
    mv /tmp/gpu.output $OUTPUTDIR

    kill $pid
    wait $pid
}

do_jct $LOGDIR/conv25 test_tf.test_mnist_tf TestMnistConv test_rpc_0 test_gpu_0
do_jct $LOGDIR/conv50 test_tf.test_mnist_tf TestMnistConv test_rpc_1 test_gpu_1
do_jct $LOGDIR/conv100 test_tf.test_mnist_tf TestMnistConv test_rpc_2 test_gpu_2

do_jct $LOGDIR/mnist25 test_tf.test_mnist_tf TestMnistLarge test_rpc_0 test_gpu_0
do_jct $LOGDIR/mnist50 test_tf.test_mnist_tf TestMnistLarge test_rpc_1 test_gpu_1
do_jct $LOGDIR/mnist100 test_tf.test_mnist_tf TestMnistLarge test_rpc_2 test_gpu_2

do_jct $LOGDIR/vgg25 test_tf.test_vgg TestVgg16 test_rpc_only_0 test_gpu_0
do_jct $LOGDIR/vgg50 test_tf.test_vgg TestVgg16 test_rpc_only_1 test_gpu_1
do_jct $LOGDIR/vgg100 test_tf.test_vgg TestVgg16 test_rpc_only_2 test_gpu_2

do_jct $LOGDIR/res25 test_tf.test_resnet TestResNetFakeData test_rpc_0 test_gpu_0
do_jct $LOGDIR/res50 test_tf.test_resnet TestResNetFakeData test_rpc_1 test_gpu_1
do_jct $LOGDIR/res75 test_tf.test_resnet TestResNetFakeData test_rpc_2 test_gpu_2

do_jct $LOGDIR/ptbT test_tf.test_seq TestSeqPtb test_rpc_0_tiny test_gpu_0_tiny
do_jct $LOGDIR/ptbS test_tf.test_seq TestSeqPtb test_rpc_1_small test_gpu_1_small
do_jct $LOGDIR/ptbM test_tf.test_seq TestSeqPtb test_rpc_2_medium test_gpu_2_medium
do_jct $LOGDIR/ptbL test_tf.test_seq TestSeqPtb test_rpc_3_large test_gpu_3_large
