# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import init_ops


def dynamicMLP(mlp_input_tensor, condition_input_tensor, output_num_units=128):
    """
    param mlp_input_tensor: mlp input (?, dim_mlp)
    param condition_input_tensor: condition input (?, dim_condition)
    param output_num_units: output dim_out
    return: (?, dim_out)
    """

    input_num_units = mlp_input_tensor.get_shape()[1]  # dim_mlp

    with tf.name_scope("dynamic_public"):
        # public (dim_mlp, dim_out) (dim_out,)
        dynamic_public_weight = tf.get_variable(
            name='dynamic_public_weight',
            dtype=tf.float32,
            shape=(input_num_units, output_num_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        dynamic_public_bias = tf.get_variable(
            name='dynamic_public_bias',
            dtype=tf.float32,
            shape=(output_num_units,),
            initializer=init_ops.glorot_uniform_initializer()
        )

    with tf.name_scope("dynamic_condition"):
        # dim_condition
        dynamic_condition_input_num_units = condition_input_tensor.get_shape()[1]
        # simple 1 layer testing
        # matrix decomposition online
        # dim_condition_out = dim_mlp * dim_out + dim_out
        dynamic_condition_output_num_units = input_num_units * output_num_units + output_num_units
        dynamic_condition_generate_weight = tf.get_variable(  # (dim_condition, dim_condition_out)
            name='dynamic_condition_weight',
            dtype=tf.float32,
            shape=(dynamic_condition_input_num_units, dynamic_condition_output_num_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        # (?, dim_condition) (dim_condition, dim_condition_out) -> (?, dim_condition_out)
        dynamic_condition_output = tf.matmul(condition_input_tensor, dynamic_condition_generate_weight)
        dynamic_condition_output = tf.nn.sigmoid(dynamic_condition_output)

    with tf.name_scope("dynamic_private"):
        # private (?, dim_mlp, dim_out) (?, dim_out)
        dynamic_private_weight = tf.reshape(dynamic_condition_output[:, 0:input_num_units * output_num_units],
                                            [-1, input_num_units, output_num_units])
        dynamic_private_bias = tf.reshape(dynamic_condition_output[:,
                                          input_num_units * output_num_units:input_num_units * output_num_units + output_num_units],
                                          [-1, output_num_units])

    with tf.name_scope("dynamic_union"):
        # union
        # (dim_mlp, dim_out) * (?, dim_mlp, dim_out) -> (?, dim_mlp, dim_out)
        # (dim_out,) * (?, dim_out) -> (?, dim_out)
        dynamic_union_weight = tf.multiply(dynamic_public_weight, dynamic_private_weight)
        dynamic_union_bias = tf.multiply(dynamic_public_bias, dynamic_private_bias)

    with tf.name_scope("dynamic_output"):
        # (?, 1, dim_mlp) (?, dim_mlp, dim_out) -> (?, 1, dim_out) -> (?, dim_out)
        input_tensor_expand = tf.expand_dims(mlp_input_tensor, axis=1)
        output = tf.matmul(input_tensor_expand, dynamic_union_weight)
        output_reduce = tf.reduce_mean(output, axis=1)
        dynamic_output = tf.nn.relu(output_reduce + dynamic_union_bias)

    return dynamic_output


def dynamicGate(gate_input_tensor, out_units):
    """
        param gate_input_tensor: gate input (?, dim_gate)
        return: (?, out_units)
    """
    with tf.name_scope("dynamic_gate"):
        # dim_gate
        dynamic_gate_input_num_units = gate_input_tensor.get_shape()[1]
        # dim_gate_out = 1
        dynamic_gate_output_num_units = out_units
        dynamic_gate_generate_weight = tf.get_variable(  # (dim_gate, dim_gate_out)
            name='dynamic_gate_weight',
            dtype=tf.float32,
            shape=(dynamic_gate_input_num_units, dynamic_gate_output_num_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        dynamic_gate_generate_bias = tf.get_variable(  # (dim_gate, dim_gate_out)
            name='dynamic_gate_bias',
            dtype=tf.float32,
            shape=(dynamic_gate_output_num_units, ),
            initializer=init_ops.glorot_uniform_initializer()
        )
        # (?, dim_gate) (dim_gate, dim_gate_out) -> (?, dim_gate_out)
        dynamic_gate_output = tf.matmul(gate_input_tensor, dynamic_gate_generate_weight)
        dynamic_gate_output = tf.nn.sigmoid(dynamic_gate_output + dynamic_gate_generate_bias)

        return dynamic_gate_output


def dynamic_union(similarity_input, diversity_input, condition_input, output_num_units):
    """
    Args:
        similarity_input:
        diversity_input:
        condition_input:
        output_num_units:
    Returns:
        union_output (?, output_num_units)
        gate_output  (?, 1)
    """
    with tf.variable_scope("similarity"):
        similarity_input = tf.concat([similarity_input], axis=-1)
        similarity_mlp_weight = tf.get_variable(
            name='similarity_mlp_weight',
            dtype=tf.float32,
            shape=(similarity_input.get_shape().as_list()[-1], output_num_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        aggregated_similarity = tf.matmul(similarity_input, similarity_mlp_weight)

    with tf.variable_scope("diversity"):
        diversity_mlp_weight = tf.get_variable(
            name='diversity_mlp_weight',
            dtype=tf.float32,
            shape=(diversity_input.get_shape().as_list()[-1], output_num_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        aggregated_diversity = tf.matmul(diversity_input, diversity_mlp_weight)

    # deep union
    mlp_input = tf.concat([aggregated_similarity, aggregated_diversity], axis=-1)
    deep_union_output = dynamicMLP(mlp_input, condition_input, output_num_units)

    # shallow union
    gate_output = dynamicGate(condition_input, output_num_units)
    shallow_union_output = gate_output * aggregated_diversity + (1 - gate_output) * aggregated_similarity

    return tf.concat([deep_union_output, shallow_union_output], axis=-1)


    

