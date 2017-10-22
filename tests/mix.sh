#! /bin/sh

export CUDA_VISIBLE_DEVICES=0,1
export EXEC_ITER_NUMBER=10
rm -f *.output

COMMANDS=(
    "python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --batch_size=25 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet50_25.output 2>&1 &"
    "python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --batch_size=50 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet50_50.output 2>&1 &"
    "python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --batch_size=75 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet50_75.output 2>&1 &"
    "python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet101 --batch_size=25 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet101_25.output 2>&1 &"
    "python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet101 --batch_size=50 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet101_50.output 2>&1 &"
    #"python ../../tf_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet101 --batch_size=75 --local_parameter_device=cpu --num_gpus=1  --variable_update=parameter_server --nodistortions --display_every=1 > resu_mix_resnet101_75.output 2>&1 &"
    #"python -m test_tf.test_vgg TestVgg16.test_rpc_only_0 > vgg16_25.output 2>&1 &"
    #"python -m test_tf.test_vgg TestVgg16.test_rpc_only_1 > vgg16_50.output 2>&1 &"
    #"python -m test_tf.test_vgg TestVgg16.test_rpc_only_2 > vgg16_100.output 2>&1 &"

    #"python -m test_tf.test_vgg TestVgg19.test_rpc_only_0 > vgg19_25.output 2>&1 &"
    #"python -m test_tf.test_vgg TestVgg19.test_rpc_only_1 > vgg19_50.output 2>&1 &"
    #"python -m test_tf.test_vgg TestVgg19.test_rpc_only_2 > vgg19_100.output 2>&1 &"

#    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_0 > mnist25.output 2>&1 &"
#    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_1 > mnist50.output 2>&1 &"
#    "python -m test_tf.test_mnist_tf TestMnistLarge.test_rpc_2 > mnist100.output 2>&1 &"
)

SHUFFLED=$(shuf -e "${COMMANDS[@]}")
readarray -t cmds <<<"$SHUFFLED"

for (( i = 0; i < ${#cmds[@]} ; i++ )); do
    cmd=${cmds[$i]}
    printf "\n**** Running: ${cmd} *****\n\n"
    # Run each command in array
    eval "stdbuf -o0 -e0 -- ${cmd}"
done
wait

