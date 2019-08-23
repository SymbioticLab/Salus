#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1
rm -f *.output

JOBS=${1:-5}
for i in $(seq $JOBS); do
    echo $i
    #python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_1 > $i.output 2>&1 &
    export SALUS_TIMEOUT=0
    python -m test_tf.test_vgg TestVgg16.test_rpc_only_2 > $i.output 2>&1 &
    #sleep 1
done
wait
exit

python -m test_tf.test_vgg TestVgg16.test_rpc_only_2 > vgg.output 2>&1 &
sleep 1
python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_1 > mnist.output 2>&1 &
wait
exit


for i in $(seq $JOBS); do
    echo $i
    #python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc > $i.output 2>&1 &
    python -m test_tf.test_vgg TestVgg16.test_rpc_only > $i.output 2>&1 &
done
wait
exit


