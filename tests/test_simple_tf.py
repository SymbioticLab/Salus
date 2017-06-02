import tensorflow as tf

with tf.device('/device:RPC:0'):
    a = tf.constant(1, name='const_1')
    b = tf.constant(2, name='const_2')
    c = tf.multiply(a, b, name='mul_1_2')

sess = tf.Session()
print(sess.run(c))
