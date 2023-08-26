# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers


class EncoderLayer(object):
    def __init__(self, num_heads=4, num_units=256, output_num_units=128, rate=0.1):
        self.num_heads = num_heads
        self.num_units = num_units
        self.output_num_units = output_num_units
        self.dropout_rate = rate

    def call(self, q, k, v, mask, training, attention_type='self_attention'):
        """
        :param q: (?, seq_len_q, d_q)
        :param k: (?, seq_len_k, d_k)
        :param v: (?, seq_len_k, d_k)
        :param mask: (?, seq_len_k)
        :param training:
        :param attention_type: self_attention || target_attention
        :return: (?, seq_len_q, output_num_units)
        """
        mha = MultiHeadAttention(self.num_heads, self.num_units, self.output_num_units)

        # (?, seq_len_q, output_num_units)
        attn_output, attention_weights = mha.call(q, k, v, mask, attention_type)

        # (batch_size, seq_len_q, output_num_units)
        ffn_output = point_wise_feed_forward_network(attn_output, self.output_num_units)
        ffn_output = tf.layers.dropout(ffn_output, rate=self.dropout_rate, training=training)

        # (batch_size, seq_len_q, output_num_units)
        out = attn_output + ffn_output

        return out, attention_weights


class MultiHeadAttention(object):
    def __init__(self, num_heads, num_units, output_num_units):
        self.num_heads = num_heads
        self.num_units = num_units
        self.output_num_units = output_num_units

        assert num_units % num_heads == 0
        assert output_num_units % num_heads == 0

        self.depth = num_units // num_heads
        self.output_depth = output_num_units // num_heads

    def split_heads(self, x, depth):
        """
        x [?, seq_len, num_units] num_heads*depth = num_units
        """
        # [?, seq_len, input_num_units] -> [?, seq_len, num_head, depth]
        x = tf.reshape(x, (-1, tf.shape(x)[1], self.num_heads, depth))
        # [?, num_head, seq_len, depth]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, attention_type='self_attention'):
        """
        :param q: (?, seq_len_q, d_q)
        :param k: (?, seq_len_k, d_k)
        :param v: (?, seq_len_k, d_k)
        :param mask: (?, seq_len_k)
        :param attention_type: self_attention || target_attention
        :return: (?, seq_len_q, output_num_units)
        """

        # Linear projections
        with tf.name_scope("linear_projection"):
            # (?, seq_len_q, num_units)
            q = layers.fully_connected(q, self.num_units, activation_fn=tf.nn.relu)
            # (?, seq_len_k, num_units)
            k = layers.fully_connected(k, self.num_units, activation_fn=tf.nn.relu)
            # (?, seq_len_k, output_num_units)
            v = layers.fully_connected(v, self.output_num_units, activation_fn=tf.nn.relu)

        with tf.name_scope("split_heads"):
            q = self.split_heads(q, self.depth)         # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, self.depth)         # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, self.output_depth)  # (batch_size, num_heads, seq_len_k, output_depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, output_depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        with tf.name_scope("scaled_dot_product_attention"):
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v,
                                                                               mask,
                                                                               self.num_heads,
                                                                               need_qk_ln=True,
                                                                               atten_type=attention_type)

        # (?, num_heads, seq_len_q, output_depth) -> (?, seq_len_q, num_heads, output_depth) -> (?, seq_len_q, output_num_units)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention, (-1, tf.shape(q)[-2], self.output_num_units))

        # (?, seq_len_q, output_num_units)
        return scaled_attention, attention_weights


def scaled_dot_product_attention(q, k, v,
                                 mask=None,
                                 num_heads=1,
                                 need_qk_ln=True,
                                 atten_type='self_attention'
                                 ):
    """
    """
    seq_len_q = tf.shape(q)[-2]
    seq_len_k = tf.shape(k)[-2]
    if mask is not None:
        mask = tf.cast(mask, tf.bool)

    with tf.name_scope("qk_ln"):
        if need_qk_ln:
            q = layers.layer_norm(q, begin_norm_axis=-1, begin_params_axis=-1)
            k = layers.layer_norm(k, begin_norm_axis=-1, begin_params_axis=-1)

    with tf.name_scope("matmul_qk"):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)  # (..., seq_len_q, seq_len_k)

    with tf.name_scope("key_mask"):
        if mask is not None:
            # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k) -> (batch_size, num_heads, seq_len_q, seq_len_k)
            key_masks = tf.reshape(mask, [-1, 1, 1, seq_len_k])
            key_masks = tf.tile(key_masks, [1, num_heads, seq_len_q, 1])
            paddings = tf.fill(tf.shape(key_masks), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
            scaled_attention_logits = tf.where(key_masks, scaled_attention_logits, paddings)

    with tf.name_scope("softmax"):
        attention_weights = tf.nn.softmax(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    with tf.name_scope("query_mask"):
        if atten_type == "self_attention":
            query_masks = mask   # (batch_size, seq_len_k)
            if query_masks is not None:
                # (batch_size, 1, seq_len_k, 1) -> (batch_size, num_head, seq_len_k, seq_len_k)
                query_masks = tf.reshape(query_masks, [-1, 1, seq_len_k, 1])
                query_masks = tf.tile(query_masks, [1, num_heads, 1, seq_len_k])
                paddings = tf.fill(tf.shape(query_masks), tf.constant(0, dtype=tf.float32))
                attention_weights = tf.where(query_masks, attention_weights, paddings)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, output_depth)

    return output, attention_weights


def point_wise_feed_forward_network(inputs, num_units):
    # (?, seq_len, num_units*4)
    outputs = layers.fully_connected(inputs, num_units * 4, activation_fn=tf.nn.relu)
    # (?, seq_len, num_units)
    outputs = layers.fully_connected(outputs, num_units, activation_fn=None)

    return outputs
