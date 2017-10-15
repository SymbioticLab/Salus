#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1
rm -f *.output

COMMANDS=(
    "python -m test_tf.test_vgg TestVgg16.test_rpc_only_0 > vgg25.output 2>&1 &"
    "python -m test_tf.test_vgg TestVgg16.test_rpc_only_1 > vgg50.output 2>&1 &"
    "python -m test_tf.test_vgg TestVgg16.test_rpc_only_2 > vgg100.output 2>&1 &"

    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_0 > mnist25.output 2>&1 &"
    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_1 > mnist50.output 2>&1 &"
    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_2 > mnist100.output 2>&1 &"
)

SHUFFLED=$(shuf -e "${COMMANDS[@]}")
readarray -t cmds <<<"$SHUFFLED"

for (( i = 0; i < ${#cmds[@]} ; i++ )); do
    cmd=${cmds[$i]}
    printf "\n**** Running: ${cmd} *****\n\n"
    # Run each command in array
    eval "${cmd}"
done
wait

