# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import init_ops


def din(q, k, v, key_masks):
    """
      q:         (?, dim)
      k:         (?, seq_len, dim)
      v:         (?, seq_len, dim_v)
      key_mask:  (?, seq_len)
      Returns
      output:    (?, dim_v)
    """
    dim_v = v.get_shape().as_list()[-1]
    seq_len = k.get_shape().as_list()[1]
    q_expand = tf.expand_dims(q, 1)               # (?, 1, dim)
    q_tile = tf.tile(q_expand, [1, seq_len, 1])   # (?, seq_len, dim)

    din_input = tf.concat([q_tile, k, q_tile - k, q_tile * k], axis=-1)  # (?, seq_len, dim * 4)

    d_layer_1_all = tf.layers.dense(din_input, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, seq_len])  # (?, seq_len, 1) -> (?, 1, seq_len)
    outputs = d_layer_3_all                           # (?, 1, seq_len)

    # Mask
    if key_masks is not None:
        key_masks = tf.expand_dims(key_masks, 1)          # [?, 1, seq_len]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [?, 1, seq_len]

    # Scale
    outputs = outputs / (k.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)                  # [?, 1, seq_len]
    alphas = outputs

    # Weighted sum (?, 1, seq_len) * (?, seq_len, dim_v) -> (?, 1, dim_v)
    outputs = tf.matmul(alphas, v)
    outputs = tf.squeeze(outputs, axis=1)
    outputs = tf.reshape(outputs, [-1, dim_v])

    return outputs, alphas


def din_wo_softmax(q, k, v, key_masks):
    """
      q:         (?, dim)
      k:         (?, seq_len, dim)
      v:         (?, seq_len, dim_v)
      key_mask:  (?, seq_len)/None
      Returns
      output:    (?, dim_v)
      alphas:    (?, 1, seq_len)
    """
    dim_v = v.get_shape().as_list()[-1]
    seq_len = k.get_shape().as_list()[1]
    q_expand = tf.expand_dims(q, 1)               # (?, 1, dim)
    q_tile = tf.tile(q_expand, [1, seq_len, 1])   # (?, seq_len, dim)

    din_input = tf.concat([q_tile, k, q_tile - k, q_tile * k], axis=-1)  # (?, seq_len, dim * 4)

    mlp_input_units = din_input.get_shape().as_list()[-1]
    mlp_output_units = 1
    din_mlp_weight = tf.get_variable(  # share
        name='din_mlp_weight',
        dtype=tf.float32,
        shape=(mlp_input_units, mlp_output_units),
        initializer=init_ops.glorot_uniform_initializer()
    )
    din_mlp_bias = tf.get_variable(    # share
        name='din_mlp_bias',
        dtype=tf.float32,
        shape=(mlp_output_units,),
        initializer=init_ops.glorot_uniform_initializer()
    )
    # (?, seq_len, dim*4) (dim*4, 1)
    mlp_output = tf.tensordot(din_input, din_mlp_weight, axes=1) + din_mlp_bias  # (?, seq_len, 1)
    mlp_output = tf.reshape(mlp_output, [-1, mlp_output_units, seq_len])         # (?, 1, seq_len)
    outputs = tf.nn.sigmoid(mlp_output)                                          # (?, 1, seq_len)

    # Mask
    if key_masks is not None:
        key_masks = tf.expand_dims(key_masks, 1)          # [?, 1, seq_len]
        paddings = tf.zeros_like(outputs)
        outputs = tf.where(key_masks, outputs, paddings)  # [?, 1, seq_len]

    alphas = outputs

    # Weighted sum (?, 1, seq_len) * (?, seq_len, dim_v) -> (?, 1, dim_v)
    outputs = tf.matmul(alphas, v)
    outputs = tf.squeeze(outputs, axis=1)
    outputs = tf.reshape(outputs, [-1, dim_v])

    return outputs, alphas


def din_attribute_attention_value(q, k, key_masks):
    """
      q:         (?, dim_q = num_a * dim_a)
      k:         (?, seq_len, dim_k = num_a * dim_a)
      key_mask:  (?, seq_len) / None
      Returns
      alphas:    (?, seq_len, num_a)
    """
    dim_q = q.get_shape().as_list()[-1]
    dim_k = k.get_shape().as_list()[-1]
    assert dim_q == dim_k

    dim_a = 16
    num_a = dim_q // dim_a

    seq_len = k.get_shape().as_list()[1]
    q_expand = tf.expand_dims(q, 1)                                              # (?, 1, dim_q)
    q_tile = tf.tile(q_expand, [1, seq_len, 1])                                  # (?, seq_len, dim_q)

    q_attributes = tf.reshape(q_tile, [-1, seq_len, num_a, dim_a])               # (?, seq_len, num_a, dim_q)
    k_attributes = tf.reshape(k, [-1, seq_len, num_a, dim_a])                    # (?, seq_len, num_a, dim_q)

    # (?, seq_len, num_a, dim_a * 4)
    din_input = tf.concat([q_attributes, k_attributes, q_attributes - k_attributes, q_attributes * k_attributes], axis=-1)
    din_input_reshape = tf.reshape(din_input, [-1, seq_len, num_a * dim_a * 4])  # (?, seq_len, num_a * dim_a * 4)
    din_input_split = tf.split(din_input_reshape, num_a, axis=-1)                # [(?, seq_len, dim_a * 4),...]

    mlp_input_units = din_input.get_shape().as_list()[-1]
    mlp_output_units = 1
    din_output_split = []
    for i in range(num_a):
        din_input_attribute = din_input_split[i]  # (?, seq_len, dim_a * 4)
        din_mlp_weight = tf.get_variable(         # share in same attribute
            name='din_mlp_weight_' + str(i),
            dtype=tf.float32,
            shape=(mlp_input_units, mlp_output_units),
            initializer=init_ops.glorot_uniform_initializer()
        )
        din_mlp_bias = tf.get_variable(
            name='din_mlp_bias_' + str(i),
            dtype=tf.float32,
            shape=(mlp_output_units,),
            initializer=init_ops.glorot_uniform_initializer()
        )
        # (?, seq_len, dim_a * 4) (dim_a * 4, 1)
        mlp_output = tf.tensordot(din_input_attribute, din_mlp_weight, axes=1) + din_mlp_bias  # (?, seq_len, 1)
        mlp_output = tf.reshape(mlp_output, [-1, seq_len, 1])
        mlp_output = tf.nn.sigmoid(mlp_output)                                                 # (?, seq_len, 1)
        din_output_split.append(mlp_output)                                                    # [(?, seq_len, 1),...]
    din_outputs = tf.concat(din_output_split, axis=-1)                                         # (?, seq_len, num_a)

    # Mask
    if key_masks is not None:
        key_masks = tf.expand_dims(key_masks, -1)                  # (?, seq_len, 1)
        key_masks = tf.tile(key_masks, [1, 1, num_a])              # (?, seq_len, num_a)
        paddings = tf.zeros_like(din_outputs)
        din_outputs = tf.where(key_masks, din_outputs, paddings)   # (?, seq_len, num_a)

    alphas = din_outputs

    return alphas


def din_user_interest_emb(trigger_alphas, target_alphas, user_seq):
    """
    Args:
        trigger_alphas: (?, seq_len, num_a)
        target_alphas:  (?, seq_len, num_a)
        user_seq:       (?, seq_len, dim_v)
    Returns:
        TA_user_interest:   (?, dim_v)
        TT_user_interest: (?, dim_v)
    """
    dim_v = user_seq.get_shape().as_list()[-1]

    TTw = trigger_alphas * target_alphas
    TAw = target_alphas

    avg_TTw = tf.reduce_mean(TTw, axis=-1, keepdims=True)      # (?, seq_len, 1)
    avg_TAw = tf.reduce_mean(TAw, axis=-1, keepdims=True)      # (?, seq_len, 1)
    avg_TTw = tf.transpose(avg_TTw, [0, 2, 1])                 # (?, 1, seq_len)
    avg_TAw = tf.transpose(avg_TAw, [0, 2, 1])                 # (?, 1, seq_len)

    # (?, 1, seq_len) * (?, seq_len, dim_v) -> (?, 1, dim_v) -> (?, dim_v)
    TT_user_interest = tf.matmul(avg_TTw, user_seq)
    TT_user_interest = tf.squeeze(TT_user_interest, axis=1)
    TT_user_interest = tf.reshape(TT_user_interest, [-1, dim_v])

    TA_user_interest = tf.matmul(avg_TAw, user_seq)
    TA_user_interest = tf.squeeze(TA_user_interest, axis=1)
    TA_user_interest = tf.reshape(TA_user_interest, [-1, dim_v])

    return TA_user_interest, TT_user_interest


def din_item_emb(trigger_alphas, target_alphas, target_input, trigger_input):
    """
    Args:
        trigger_alphas: (?, seq_len, num_a)
        target_alphas:  (?, seq_len, num_a)
        target_input:   (?, num_a * dim_a)
        trigger_input:  (?, num_a * dim_a)
    Returns:
        TA_item_emb:    (?, dim)
        TT_item_emb:  (?, dim * 2)
    """
    dim = target_input.get_shape().as_list()[-1]
    dim_a = 16
    num_a = dim // dim_a
    target_input = tf.reshape(target_input, [-1, num_a, dim_a])     # (?, num_a, dim_a)
    trigger_input = tf.reshape(trigger_input, [-1, num_a, dim_a])   # (?, num_a, dim_a)
    cross_input = tf.concat([target_input - trigger_input, target_input * trigger_input], axis=-1)  # (?, num_a, dim_a * 2)

    TTw = trigger_alphas * target_alphas
    TAw = target_alphas
    avg_TTw = tf.reduce_mean(TTw, axis=1, keepdims=True)      # (?, 1, num_a)
    avg_TAw = tf.reduce_mean(TAw, axis=1, keepdims=True)      # (?, 1, num_a)

    # (?, 1, num_a) (?, num_a, dim_a * 2) -> (?, 1, dim_a * 2) -> (?, dim_a * 2)
    TT_item_emb = tf.matmul(avg_TTw, cross_input)
    TT_item_emb = tf.squeeze(TT_item_emb, axis=1)
    TT_item_emb = tf.reshape(TT_item_emb, [-1, dim_a * 2])

    TA_item_emb = tf.matmul(avg_TAw, target_input)
    TA_item_emb = tf.squeeze(TA_item_emb, axis=1)
    TA_item_emb = tf.reshape(TA_item_emb, [-1, dim_a])

    return TA_item_emb, TT_item_emb


def dpan_din_output(trigger_total, target_total, user_seq, seq_mask):
    """
    Args:
        trigger_total: (?, dim)
        target_total:  (?, dim)
        user_seq:      (?, seq_len, dim)
        seq_mask:      (?, seq_len)
    Returns:
        similarity_input (?, dim_a * 2 + dim_v)
        diversity_input  (?, dim_a + dim_v)
        aux_alphas       (?, num_a)
    """
    # --------- AAVG -----------
    # (?, seq_len, num_a)
    trigger_alphas = din_attribute_attention_value(trigger_total, user_seq, seq_mask)
    target_alphas = din_attribute_attention_value(target_total, user_seq, seq_mask)
    # (?, 1, num_a) -> (?, num_a)
    aux_alphas = din_attribute_attention_value(target_total, tf.expand_dims(trigger_total, 1), None)
    aux_alphas = tf.squeeze(aux_alphas, 1)

    # --------- BCR -----------
    diversity_user_interest, similarity_user_interest = din_user_interest_emb(trigger_alphas, target_alphas, user_seq)
    diversity_item_emb, similarity_item_emb = din_item_emb(trigger_alphas, target_alphas, target_total, trigger_total)

    similarity_input = tf.concat([similarity_user_interest, similarity_item_emb], axis=-1)
    diversity_input = tf.concat([diversity_user_interest, diversity_item_emb], axis=-1)

    return similarity_input, diversity_input, aux_alphas
