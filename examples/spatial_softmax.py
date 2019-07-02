import tensorflow as tf
tf.enable_eager_execution()

N, H, W, C = 1, 2, 4, 3
features = tf.get_variable('features', shape=[N, H, W, C], initializer=tf.random_normal_initializer())
print(features)

# transpose features [N, H, W, C] to [N, C, H, W]
features_t = tf.transpose(features, [0, 3, 1, 2])
print(features_t)

# reshape features transpose to [N*C, H*W]
features_tr = tf.reshape(features_t, [N*C, H*W])
print(features_tr)

# softmax
softmax = tf.nn.softmax(features_tr)
print('\n\n--------softmax--------')
print(softmax)

'''
# reshape and retranspose
softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])
'''

pos_x, pos_y = tf.meshgrid(tf.lin_space(-1., 1., num=H),
                         tf.lin_space(-1., 1., num=W),
                         indexing='ij')
print('\n\n--------posx--------')
print(pos_x)
print('\n\n--------posy--------')
print(pos_y)

pos_x = tf.reshape(pos_x, [H * W])
print(pos_x)

pos_y = tf.reshape(pos_y, [H * W])
print(pos_y)

expected_x = tf.reduce_sum(softmax * pos_x, axis=1, keepdims=True)
expected_y = tf.reduce_sum(softmax * pos_y, axis=1, keepdims=True)

expected_xy = tf.concat([expected_x, expected_y], axis=1)
print(expected_xy)

flat_expected_xy = tf.reshape(expected_xy, [-1, C * 2])
print('\n\n---------flat--------')
print(flat_expected_xy)