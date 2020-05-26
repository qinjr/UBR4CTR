import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
import gc
import numpy as np
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

def expand(x, axis, N, dims=2):
    if dims != 2:
        return tf.tile(tf.expand_dims(x, axis), [N, 1, 1])
    return tf.tile(tf.expand_dims(x, axis), [N, 1])
    # return tf.concat([tf.expand_dims(x, dim) for _ in tf.range(N)], axis=dim)


def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(
        tf.ones([1, 1]), units, activation_fn=None, biases_initializer=None))


class MIMNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_units, memory_vector_dim, batch_size=128, memory_size=4,
                 read_head_num=1, write_head_num=1, reuse=False, output_dim=16, clip_value=20, sharp_value=2.):
        self.controller_units = controller_units
        self.memory_vector_dim = memory_vector_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.reuse = reuse
        self.clip_value = clip_value
        self.sharp_value = sharp_value

        def single_cell(num_units):
            return tf.nn.rnn_cell.GRUCell(num_units)

        self.controller = single_cell(self.controller_units)
        self.step = 0
        self.output_dim = output_dim

        # TODO: ?
        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(
            self.controller_units + self.memory_vector_dim * self.read_head_num)

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state["controller_state"])

        num_parameters_per_head = self.memory_vector_dim + 1  # TODO: why +1? sharp_value?
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num

        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(
                controller_output, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)
            parameters = tf.clip_by_norm(parameters, self.clip_value)

        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        prev_M = prev_state["M"]
        key_M = prev_state["key_M"]
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = (tf.nn.softplus(head_parameter[:, self.memory_vector_dim]) + 1) * self.sharp_value
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, key_M, prev_M)  # [batch_size, memory_size]
            w_list.append(w)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            # [batch_size, fnum * eb_dim]
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector = tf.reshape(read_vector, [-1, self.memory_vector_dim])
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]

        M = prev_M
        sum_aggre = prev_state["sum_aggre"]

        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)  # [batch_size, memory_size, 1]
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)  # [batch_size, 1, fnum * eb_dim]
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)

            # [batch_size, memory_size, fnum * eb_dim]
            # M_t = (1 - E_t) * M_t + A_t
            ones = tf.ones([self.batch_size, self.memory_size, self.memory_vector_dim])
            M = M * (ones - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)
            sum_aggre += tf.matmul(tf.stop_gradient(w), add_vector)  # [batch_size, memory_size, fnum * eb_dim]

        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            read_output = tf.contrib.layers.fully_connected(
                tf.concat([controller_output] + read_vector_list, axis=1), self.output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)
            read_output = tf.clip_by_norm(read_output, self.clip_value)

        self.step += 1
        return read_output, {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "w_list": w_list,
            "M": M,
            "key_M": key_M,
            "sum_aggre": sum_aggre
        }

    def addressing(self, k, beta, key_M, prev_M):
        # Cosine Similarity
        def cosine_similarity(key, M):
            key = tf.expand_dims(key, axis=2)
            inner_product = tf.matmul(M, key)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1, keep_dims=True))
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))
            return K

        K = 0.5*(cosine_similarity(k,key_M) + cosine_similarity(k,prev_M))
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)

        return w_c

    def zero_state(self, batch_size):
        with tf.variable_scope('init', reuse=self.reuse):
            read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim)), 0, batch_size)
                                for _ in range(self.read_head_num)]

            w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), 0, batch_size)
                      for _ in range(self.read_head_num + self.write_head_num)]

            controller_init_state = self.controller.zero_state(batch_size, tf.float32)

            M = expand(tf.tanh(tf.get_variable(
                'init_M', [self.memory_size, self.memory_vector_dim],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-5), trainable=False)), 0, batch_size, 3)

            key_M = expand(tf.tanh(tf.get_variable(
                    'key_M', [self.memory_size, self.memory_vector_dim],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), 0, batch_size, 3)

            sum_aggre = tf.zeros([batch_size, self.memory_size, self.memory_vector_dim], dtype=tf.float32)

            state = {
                "controller_state": controller_init_state,
                "read_vector_list": read_vector_list,
                "w_list": w_list,
                "M": M,
                "key_M": key_M,
                "sum_aggre": sum_aggre
            }
            return state

class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = core_rnn_cell._Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = core_rnn_cell._Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h