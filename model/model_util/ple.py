# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import init_ops


def multi_task_ple(
        task_input,
        task_shared_input,
        task_num,
        num_level,
        experts_units,
        experts_num):
    """
    :param task_input: input tensor
    :param task_shared_input: input tensor
    :param task_num:
    :param num_level: extraction
    :param experts_units:
    :param experts_num:
    :return:
    """
    gate_output_task_final_list = [task_input for _ in range(task_num)]
    gate_output_alpha_list = [None for _ in range(task_num)]
    gate_output_shared_final = task_shared_input
    selector_num = 2
    shared_selector_num = task_num + 1

    for i in range(num_level):
        with tf.name_scope("PLE_LEVEL_%d" % i):
            with tf.name_scope("EXPERT_WEIGHT_INITIAL"):
                # experts shared
                experts_weight = tf.get_variable(
                    name='experts_weight_%d' % i,
                    dtype=tf.float32,
                    shape=(gate_output_shared_final.get_shape()[1], experts_units, experts_num),
                    initializer=init_ops.glorot_uniform_initializer()
                )

                experts_bias = tf.get_variable(
                    name='expert_bias_%d' % i,
                    dtype=tf.float32,
                    shape=(experts_units, experts_num),
                    initializer=init_ops.glorot_uniform_initializer()
                )

                experts_weight_task_list = []
                experts_bias_task_list = []
                for task_index in range(task_num):
                    experts_weight_task = tf.get_variable(
                        name='experts_weight_task%d_%d' % (task_index, i),
                        dtype=tf.float32,
                        shape=(gate_output_task_final_list[task_index].get_shape()[1], experts_units, experts_num),
                        initializer=init_ops.glorot_uniform_initializer()
                    )
                    experts_weight_task_list.append(experts_weight_task)

                    experts_bias_task = tf.get_variable(
                        name='expert_bias_task%d_%d' % (task_index, i),
                        dtype=tf.float32,
                        shape=(experts_units, experts_num),
                        initializer=init_ops.glorot_uniform_initializer()
                    )
                    experts_bias_task_list.append(experts_bias_task)

            with tf.name_scope("GATE_WEIGHT_INITIAL"):
                # gates shared
                gate_shared_weight = tf.get_variable(
                    name='gate_shared_%d' % i,
                    dtype=tf.float32,
                    shape=(gate_output_shared_final.get_shape()[1], experts_num * shared_selector_num),
                    initializer=init_ops.glorot_uniform_initializer()
                )
                gate_shared_bias = tf.get_variable(
                    name='gate_shared_bias_%d' % i,
                    dtype=tf.float32,
                    shape=(experts_num * shared_selector_num,),
                    initializer=init_ops.glorot_uniform_initializer()
                )

                # gates Task
                gate_weight_task_list = []
                gate_bias_task_list = []
                for task_index in range(task_num):
                    gate_weight_task = tf.get_variable(
                        name='gate_weight_task%d_%d' % (task_index, i),
                        dtype=tf.float32,
                        shape=(gate_output_task_final_list[task_index].get_shape()[1], experts_num * selector_num),
                        initializer=init_ops.glorot_uniform_initializer()
                    )
                    gate_weight_task_list.append(gate_weight_task)
                    gate_bias_task = tf.get_variable(
                        name='gate_bias_task%d_%d' % (task_index, i),
                        dtype=tf.float32,
                        shape=(experts_num * selector_num,),
                        initializer=init_ops.glorot_uniform_initializer()
                    )
                    gate_bias_task_list.append(gate_bias_task)

            with tf.name_scope("EXPERT_OUTPUT"):
                # experts shared outputs
                experts_output = tf.tensordot(gate_output_shared_final, experts_weight, axes=1)
                experts_output = tf.add(experts_output, experts_bias)
                experts_output = tf.nn.relu(experts_output)

                # experts Task outputs
                experts_output_task_list = []
                for task_index in range(task_num):
                    experts_output_task = tf.tensordot(gate_output_task_final_list[task_index], experts_weight_task_list[task_index], axes=1)
                    experts_output_task = tf.add(experts_output_task, experts_bias_task_list[task_index])
                    experts_output_task = tf.nn.relu(experts_output_task)
                    experts_output_task_list.append(experts_output_task)

            with tf.name_scope("GATE_EXPERT_OUTPUT"):
                for task_index in range(task_num):
                    # gates Task outputs
                    gate_output_task = tf.matmul(gate_output_task_final_list[task_index], gate_weight_task_list[task_index])
                    gate_output_task = tf.add(gate_output_task, gate_bias_task_list[task_index])
                    gate_output_task = tf.nn.softmax(gate_output_task)
                    gate_output_alpha = gate_output_task
                    gate_output_alpha_list[task_index] = gate_output_alpha

                    gate_output_task = tf.multiply(
                        tf.concat([experts_output_task_list[task_index], experts_output], axis=2),
                        tf.expand_dims(gate_output_task, axis=1)
                    )
                    gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
                    gate_output_task = tf.reshape(gate_output_task, [-1, experts_units])
                    gate_output_task_final = gate_output_task
                    gate_output_task_final_list[task_index] = gate_output_task_final

                with tf.name_scope("TASK_SHARED_GATE"):
                    # gates shared outputs
                    gate_output_shared = tf.matmul(gate_output_shared_final, gate_shared_weight)
                    gate_output_shared = tf.add(gate_output_shared, gate_shared_bias)
                    gate_output_shared = tf.nn.softmax(gate_output_shared)
                with tf.name_scope("TASK_SHARED_OUTPUT"):
                    array = experts_output_task_list
                    array.insert(1, experts_output)
                    gate_output_shared = tf.multiply(
                        tf.concat(array,
                                  axis=2),
                        tf.expand_dims(gate_output_shared, axis=1)
                    )
                    gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                    gate_output_shared = tf.reshape(gate_output_shared, [-1, experts_units])
                    gate_output_shared_final = gate_output_shared

    return gate_output_task_final_list, gate_output_alpha_list
