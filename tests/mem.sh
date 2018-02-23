#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1

EXECUTOR=../build/Release/bin/executor
LD_LIBRARY_PATH=../build/Release/lib:$LD_LIBRARY_PATH

do_mem() {
    local OUTPUTDIR=$1
    mkdir -p $OUTPUTDIR

    echo "Running $2.$3"

    env CUDA_VISIBLE_DEVICES=2,3 TF_CPP_MIN_LOG_LEVEL=4 $EXECUTOR --logconf ../build/alloc.config &
    pid=$!
    stdbuf -o0 -e0 -- \
    python -m $2 $3 > $OUTPUTDIR/mem-iter.output
    kill $pid
    wait $pid
    mv /tmp/alloc.output $OUTPUTDIR/alloc.output
}

rm -f /tmp/err.output /tmp/alloc.output

do_mem ../scripts/logs/vgg11_25 test_tf.test_vgg TestVgg16.test_rpc_only_0
do_mem ../scripts/logs/vgg11_50 test_tf.test_vgg TestVgg16.test_rpc_only_1
do_mem ../scripts/logs/vgg11_100 test_tf.test_vgg TestVgg16.test_rpc_only_2

do_mem ../scripts/logs/vgg16_25 test_tf.test_vgg TestVgg16.test_rpc_only_0
do_mem ../scripts/logs/vgg16_50 test_tf.test_vgg TestVgg16.test_rpc_only_1
do_mem ../scripts/logs/vgg16_100 test_tf.test_vgg TestVgg16.test_rpc_only_2

do_mem ../scripts/logs/vgg19_25 test_tf.test_vgg TestVgg16.test_rpc_only_0
do_mem ../scripts/logs/vgg19_50 test_tf.test_vgg TestVgg16.test_rpc_only_1
do_mem ../scripts/logs/vgg19_100 test_tf.test_vgg TestVgg16.test_rpc_only_2

exit

do_mem ../scripts/logs/ptbT test_tf.test_seq TestSeqPtb.test_rpc_0_tiny
do_mem ../scripts/logs/ptbS test_tf.test_seq TestSeqPtb.test_rpc_1_small
do_mem ../scripts/logs/ptbM test_tf.test_seq TestSeqPtb.test_rpc_2_medium
do_mem ../scripts/logs/ptbL test_tf.test_seq TestSeqPtb.test_rpc_3_large

do_mem ../scripts/logs/conv25 test_tf.test_mnist_tf TestMnistConv.test_rpc_0
do_mem ../scripts/logs/conv50 test_tf.test_mnist_tf TestMnistConv.test_rpc_1
do_mem ../scripts/logs/conv100 test_tf.test_mnist_tf TestMnistConv.test_rpc_2

do_mem ../scripts/logs/mnist25 test_tf.test_mnist_tf TestMnistLarge.test_rpc_0
do_mem ../scripts/logs/mnist50 test_tf.test_mnist_tf TestMnistLarge.test_rpc_1
do_mem ../scripts/logs/mnist100 test_tf.test_mnist_tf TestMnistLarge.test_rpc_2


do_mem ../scripts/logs/res25 test_tf.test_resnet TestResNetFakeData.test_rpc_0
do_mem ../scripts/logs/res50 test_tf.test_resnet TestResNetFakeData.test_rpc_1
do_mem ../scripts/logs/res75 test_tf.test_resnet TestResNetFakeData.test_rpc_2
