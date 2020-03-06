import tensorflow as tf
import settings

opt = settings.opt
batch_size=opt.batch_size
linear_key_dim = opt.key_dim
linear_value_dim = opt.value_dim
num_heads = opt.num_heads

def build_model(model_input, state_num, is_training=True):
    x = tf.layers.conv2d(model_input, 128, 3, padding='same', activation=None, name='conv1')
    y = x
    with tf.variable_scope("rnn"):
        for i in range(state_num):
            if i == 0:
                x, corr = residual_block(x, y, 128, is_training, name='RB1', reuse=False)
            else:
                x, corr = residual_block(x, y, 128, is_training, name='RB1', reuse=True, corr=corr)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 1, 3, padding='same', activation=None, name='conv_end')

    return x


def residual_block(x, y, filter_num, is_training, name, reuse, corr=None):
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x, corr_new = non_local_block(x, 128, 128, name='non_local', reuse=reuse, corr=corr)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, filter_num, 3, padding='same', activation=None, name=name + '_a', reuse=reuse)

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filter_num, 3, padding='same', activation=None, name=name + '_b', reuse=reuse)

    x = tf.add(x, y)
    return x, corr_new

# Multi-head Attention
def non_local_block(x, filter_num, output_filter_num, name, reuse=False, corr=None):
    # linear projection then reshape
    x_theta = tf.layers.conv2d(x, filter_num, 1, padding='same', activation=None, name=name + '_theta', reuse=reuse)
    x_phi = tf.layers.conv2d(x, filter_num, 1, padding='same', activation=None, name=name + '_phi', reuse=reuse)
    x_g = tf.layers.conv2d(x, output_filter_num, 1, padding='same', activation=None, name=name + '_g', reuse=reuse, kernel_initializer=tf.zeros_initializer())

    x_theta_reshaped = tf.reshape(x_theta, [tf.shape(x_theta)[0], tf.shape(x_theta)[1] * tf.shape(x_theta)[2],
                                            tf.shape(x_theta)[3]])
    x_phi_reshaped = tf.reshape(x_phi,
                                [tf.shape(x_phi)[0], tf.shape(x_phi)[1] * tf.shape(x_phi)[2], tf.shape(x_phi)[3]])

    x_g_reshaped = tf.reshape(x_g,
                                [tf.shape(x_g)[0], tf.shape(x_g)[1] * tf.shape(x_g)[2], tf.shape(x_g)[3]])

    # split heads, dim
    x_thetas, x_phis, x_gs = _split_heads(x_theta_reshaped, x_phi_reshaped, x_g_reshaped)
    outputs = _scaled_dot_product(x_thetas, x_phis, x_gs)
    output_dim = _concat_heads(outputs)

    # split heads, seq
    x_thetas_seq, x_phis_seq, x_gs_seq = _split_heads_seq(x_theta_reshaped, x_phi_reshaped, x_g_reshaped)
    outputs_seq = _scaled_dot_product(x_thetas_seq, x_phis_seq, x_gs_seq)
    output_seq = _concat_heads_seq(outputs_seq)
    output_seq = tf.reshape(output_seq, [batch_size, -1, filter_num])

    output = tf.concat(values=[output_seq, output_dim], axis=-1)  # axis --> dim
    x_mul2_reshaped = tf.reshape(output,
                                 [tf.shape(output)[0], tf.shape(x_phi)[1], tf.shape(x_phi)[2], output_filter_num * 2])
    output_mix = tf.layers.conv2d(x_mul2_reshaped, filter_num, 1, padding='same', activation=None, name=name + '_mix_output', reuse=reuse) # axis --> dim
    # corr
    x_mul1 = output_mix[-1]
    if corr is not None:
        x_mul1 += corr

    return tf.add(x, output_mix), x_mul1

def _split_heads(q, k, v):
    def split_last_dimension_then_transpose(tensor, num_heads, dim):
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if type(v) == type(None) else v for v in t_shape]
        tensor = tf.reshape(tensor, [batch_size] + t_shape[1:-1] + [num_heads, dim // num_heads])
        return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

    qs = split_last_dimension_then_transpose(q, num_heads, linear_key_dim)
    ks = split_last_dimension_then_transpose(k, num_heads, linear_key_dim)
    vs = split_last_dimension_then_transpose(v, num_heads, linear_value_dim)
    return qs, ks, vs

def _split_heads_seq(q, k, v):
    def split_second_dimension_then_transpose(tensor, num_heads, dim):
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if type(v) == type(None) else v for v in t_shape]
        tensor = tf.reshape(tensor, [batch_size]  + [num_heads, t_shape[1] // num_heads] + [dim])
        return tf.transpose(tensor, [0, 1, 3, 2])  # [batch_size, num_heads, dim, max_seq_len]

    qs = split_second_dimension_then_transpose(q, num_heads, linear_key_dim)
    ks = split_second_dimension_then_transpose(k, num_heads, linear_key_dim)
    vs = split_second_dimension_then_transpose(v, num_heads, linear_value_dim)
    return qs, ks, vs

def _scaled_dot_product(qs, ks, vs):
    key_dim_per_head = linear_key_dim // num_heads

    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (key_dim_per_head ** 0.5)
    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)

def _concat_heads(outputs):
    def transpose_then_concat_last_two_dimenstion(tensor):
        tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if type(v) == type(None) else v for v in t_shape]
        num_heads, dim = t_shape[2:]
        return tf.reshape(tensor, [batch_size] + [t_shape[1]] + [num_heads * dim])

    return transpose_then_concat_last_two_dimenstion(outputs)

def _concat_heads_seq(outputs):
    def transpose_then_concat_last_two_dimenstion(tensor): # tensor.shape = [batch_size, num_heads, dim, seq_len]
        tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, dim, num_heads, seq_len]
        t_shape = tensor.get_shape().as_list()
        t_shape = [-1 if type(v) == type(None) else v for v in t_shape]
        num_heads, seq_len = t_shape[2:]
        return tf.reshape(tensor, [batch_size] + [t_shape[1]] + [-1]) # -1 --> num_heads * seq_len

    return transpose_then_concat_last_two_dimenstion(outputs)
