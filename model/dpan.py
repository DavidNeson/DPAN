# coding=utf-8

from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from requests.exceptions import ConnectionError
from optimizer.adagrad import SearchAdagrad
from optimizer import optimizer_ops as myopt

from model_util.ple import *
from model_util.DIN import *
from model_util.DynamicNN import *
from model_util.encoder import *

optimizer_dict = {
    "Adagrad": lambda opt_conf, global_step: SearchAdagrad(opt_conf).get_optimizer(global_step)
}


class DPAN:

    def __init__(self):
        self.model_name = "DPAN"

        # context
        self.context = None       # context
        self.logger = None
        self.config = None        # conf, self.config.get_job_config("ps_num") aop.json
        self.algo_config = None   # algo_conf.json[model_name]
        self.opts_conf = None     # algo_conf.json[model_name][optimizer]
        self.model_conf = None    # algo_conf.json[model_name][modelx]
        self.batch_size = None
        self.is_training = tf.placeholder(tf.bool, name="training")  # feed_dict training:0
        self.global_step = None
        self.global_step_reset = None
        self.global_step_add = None

        # model input
        self.feature_columns = None
        self.features = None
        self.user_column_block = []
        self.item_column_block = []
        self.trigger_column_block = []
        self.ctx_column_block = []
        self.condition_column_block = []
        self.seq_column_block = []
        self.seq_column_len = {}
        self.layer_dict = {}
        self.sequence_layer_dict = {}
        self.seq_type_list = []
        self.dpan_attention_block_name = 'dpan_attention'
        self.seq_block_name_list = []

        # ESMM main net
        self.ctr_main_net = None
        self.cvr_main_net = None

        # label
        self.label = None
        self.ctr_label = None
        self.cvr_label = None

        # logits
        self.ctr_logits = None
        self.cvr_logits = None
        self.logits = None

        # prediction
        self.ctr_prediction = None
        self.cvr_prediction = None
        self.ctr_prob = None
        self.cvr_prob = None
        self.ctcvr_prob = None

        # loss
        self.loss_op = None
        self.ctr_loss = None
        self.cvr_loss = None
        self.ctcvr_loss = None
        self.reg_loss = None
        self.aux_ctr_loss = None
        self.aux_ctr_loss_list = None

        # auc
        self.mask_ipv = None

        # middle vars
        self.dpan_attention_aux_weight_list = []

        self.metrics = {}

    def variable_scope(self, *args, **kwargs):
        kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.config.get_job_config("ps_num"),
            min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def init(self, ctx):

        self.metrics = {}

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']

        if self.model_name is None:
            self.model_name = "DPAN"

        self.user_column_block = []
        self.item_column_block = []
        self.trigger_column_block = []
        self.ctx_column_block = []
        self.condition_column_block = []
        self.seq_column_block = []
        self.seq_column_len = {}
        self.layer_dict = {}
        self.sequence_layer_dict = {}

        if self.algo_config.get('user_column_block') is not None:
            arr_block = self.algo_config.get('user_column_block').split(';', -1)
            for block in arr_block:
                if len(block) <= 0:
                    continue
                self.user_column_block.append(block)
        else:
            raise RuntimeError("user_column_block must be specified.")

        if self.algo_config.get('item_column_block') is not None:
            arr_block = self.algo_config.get('item_column_block').split(';', -1)
            for block in arr_block:
                if len(block) <= 0:
                    continue
                self.item_column_block.append(block)
        else:
            raise RuntimeError("item_column_block must be specified.")

        if self.algo_config.get('trigger_column_block') is not None:
            arr_block = self.algo_config.get('trigger_column_block').split(';', -1)
            for block in arr_block:
                if len(block) <= 0:
                    continue
                self.trigger_column_block.append(block)
        else:
            raise RuntimeError("trigger_column_block must be specified.")

        if self.algo_config.get('ctx_column_block') is not None:
            arr_block = self.algo_config.get('ctx_column_block').split(';', -1)
            for block in arr_block:
                if len(block) <= 0:
                    continue
                self.ctx_column_block.append(block)

        if self.algo_config.get('condition_column_block') is not None:
            arr_block = self.algo_config.get('condition_column_block').split(';', -1)
            for block in arr_block:
                if len(block) <= 0:
                    continue
                self.condition_column_block.append(block)

        if self.algo_config.get('seq_column_block') is not None:
            arr_block = self.algo_config.get('seq_column_block').split(';', -1)
            for block in arr_block:
                arr = block.split(':', -1)
                if len(arr) != 2:
                    continue
                if len(arr[0]) > 0:
                    self.seq_column_block.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]

        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def build_graph(self, ctx, features, feature_columns, labels):
        self.set_global_step()
        self.inference(features, feature_columns)
        self.loss_dpan(labels)
        self.optimizer(ctx, self.loss_op)
        self.predictions()

    def set_global_step(self):
        """Sets up the global step Tensor."""
        with tf.name_scope("set_global_step"):
            self.global_step = training_util.get_or_create_global_step()
            self.global_step_reset = tf.assign(self.global_step, 0)
            self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
            tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def inference(self, features, feature_columns):
        with tf.name_scope("get_label"):
            self.ctr_label = features.get("ctr_label", None)
            self.cvr_label = features.get("cvr_label", None)

        with tf.name_scope("get_batch_size"):
            self.batch_size = tf.shape(self.ctr_label)[0]

        self.feature_columns = feature_columns
        self.features = features
        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        self.dnn_layer()
        self.ctr_logits = self.logits_layer("CTR")
        self.cvr_logits = self.logits_layer("CVR")

    def embedding_layer(self, features, feature_columns):
        with self.variable_scope(name_or_scope="Embedding_Layer") as scope:
            for block_name in set(self.user_column_block +
                                  self.item_column_block +
                                  self.trigger_column_block +
                                  self.seq_column_len.values() +
                                  self.ctx_column_block +
                                  self.condition_column_block
                                  ):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.layer_dict[block_name] = layers.input_from_feature_columns(
                                                                        features,
                                                                        feature_columns=feature_columns[block_name],
                                                                        scope=scope)

        with self.variable_scope(name_or_scope="Sequence_Embedding_Layer") as scope:
            if len(self.seq_column_block) > 0:
                for block_name in self.seq_column_block:
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError("block_name:(%s) not in feature_columns for seq" % block_name)
                    max_len = 50
                    # (?, 128)
                    sequence_layer = layers.input_from_feature_columns(features,
                                                                       feature_columns[block_name], scope=scope)
                    sequence = tf.split(sequence_layer, max_len, axis=0)
                    sequence_stack = tf.stack(values=sequence, axis=1)  # batch, seq_len, dim (?,50,128)
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])  # [batch*seq_len, dim]
                    sequence_length = self.layer_dict[self.seq_column_len[block_name]]  # real length of input (?,1)
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                    sequence_stack = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                         sequence_2d, tf.zeros_like(sequence_2d)),
                                                tf.shape(sequence_stack))  # (?,50,128)
                    # (B,N,d)
                    self.sequence_layer_dict[block_name] = sequence_stack

    def sequence_layer(self):
        dpan_attention_vec_list = []
        dpan_attention_aux_weight_list = []
        for block_name in self.sequence_layer_dict.keys():
            self.seq_type_list.append(block_name)
            with self.variable_scope(name_or_scope="{}_sequence_slat_{}".format(self.model_name, block_name)):
                max_len = 50

                # sequence stack (?, seq_len, 16*8)
                sequence_stack = self.sequence_layer_dict[block_name]
                sequence_length = self.layer_dict[self.seq_column_len[block_name]]            # (?,1)
                sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)  # (?, max_len) bool

                if self.model_conf['model_hyperparameter']['need_dpan_attention']:
                    with tf.name_scope("dpan"):
                        trigger_total = self.layer_dict["trigger_id_columns"]   # (?, 8 * 16)
                        target_total = self.layer_dict["item_id_columns"]       # (?, 8 * 16)
                        condition_input = self.layer_dict["condition_columns"]
                        # (?, 2 * dim_a + dim_v) (?, dim_a + dim_v) (?, num_a)
                        # AAVG & BCR
                        similarity_input, diversity_input, aux_alphas = dpan_din_output(trigger_total, target_total, sequence_stack, sequence_mask)

                        # SDUF
                        # (?, d_out) (?, 1)
                        dpan_attention_output = dynamic_union(similarity_input, diversity_input, condition_input, 128)

                        dpan_attention_vec_list.append(dpan_attention_output)
                        dpan_attention_aux_weight_list.append(aux_alphas)  # [(?,num_a),...]

        if self.model_conf['model_hyperparameter']['need_dpan_attention']:
            dpan_attention_output = tf.concat(dpan_attention_vec_list, axis=-1)
            self.layer_dict[self.dpan_attention_block_name] = dpan_attention_output
            self.seq_block_name_list.append(self.dpan_attention_block_name)
            self.dpan_attention_aux_weight_list = dpan_attention_aux_weight_list

    # main net
    def dnn_layer(self):
        self.ple_net()

    def ple_net(self):
        joint_features = []
        for block_name in set(self.user_column_block +
                              self.item_column_block +
                              self.trigger_column_block +
                              self.ctx_column_block +
                              self.seq_block_name_list
                              ):
            if block_name not in self.layer_dict:
                raise ValueError('[joint_features, layer dict] does not has block : {}'.format(block_name))
            joint_features.append(self.layer_dict[block_name])

        joint_expert_net = tf.concat(values=joint_features, axis=-1)
        with tf.name_scope("{}_ple_network".format(self.model_name)):
            task_num = 2
            gate_output_task_final_list, gate_output_alpha_list = \
                multi_task_ple(joint_expert_net, joint_expert_net, task_num, 2, 256, 2)
            self.ctr_main_net = gate_output_task_final_list[0]
            self.cvr_main_net = gate_output_task_final_list[1]

    def logits_layer(self, name):
        with self.variable_scope(name_or_scope="{}_{}_Logits".format(self.model_name, name)):
            if "CTR" in name:
                main_net = self.ctr_main_net
                bias_weight = "ctr_bias_weight"
            elif "CVR" in name:
                main_net = self.cvr_main_net
                bias_weight = "cvr_bias_weight"

            main_logits = layers.linear(
                main_net,
                1,
                scope="main_net",
                biases_initializer=None)
            _logits = main_logits

            bias = contrib_variables.model_variable(
                bias_weight,
                shape=[1],
                initializer=tf.zeros_initializer(),
                trainable=True)

            logits = nn_ops.bias_add(_logits, bias)

            return logits

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_tmp = []
        for reg_loss in reg_losses:
            if reg_loss.name.startswith(self.model_name) or reg_loss.name.startswith('Share'):
                reg_tmp.append(reg_loss)
        self.reg_loss = tf.reduce_sum(reg_tmp)

    def loss_dpan(self, label):
        self.label = label
        ctr_logits = self.ctr_logits
        cvr_logits = self.cvr_logits
        ctr_label = self.ctr_label
        cvr_label = self.cvr_label

        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            with tf.name_scope("p_value"):
                p_ctr = tf.sigmoid(ctr_logits)
                p_cvr = tf.sigmoid(cvr_logits)
                p_ctcvr = tf.multiply(p_ctr, p_cvr)
                self.ctr_prob = p_ctr
                self.cvr_prob = p_cvr
                self.ctcvr_prob = p_ctcvr

            with tf.name_scope("ctr_loss"):
                ctr_ce_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits)
                )
                self.ctr_loss = ctr_ce_loss
            with tf.name_scope("ctcvr_loss"):
                ctcvr_ce_loss = tf.reduce_mean(
                    tf.losses.log_loss(labels=cvr_label, predictions=p_ctcvr)
                )
                self.ctcvr_loss = ctcvr_ce_loss
            with tf.name_scope("reg_loss"):
                self.reg_loss_f()

            with tf.name_scope("aux_ctr_loss"):
                aux_ctr_loss = 0
                aux_ctr_loss_list = []

                if self.model_conf['model_hyperparameter']['need_dpan_attention']:
                    for target_weight in self.dpan_attention_aux_weight_list:                    # [(?,7),..]
                        attribute_cnt = target_weight.get_shape().as_list()[-1]                  # (?,7)
                        target_weight_split = tf.split(target_weight, attribute_cnt, axis=-1)    # [(?,1),...]
                        for attribute_score in target_weight_split:
                            attribute_ctr_log_loss = tf.reduce_mean(
                                tf.losses.log_loss(labels=ctr_label, predictions=attribute_score)
                            )
                            aux_ctr_loss_list.append(attribute_ctr_log_loss)
                            aux_ctr_loss += attribute_ctr_log_loss
                self.aux_ctr_loss = aux_ctr_loss
                self.aux_ctr_loss_list = aux_ctr_loss_list

            with tf.name_scope("mask"):
                self.mask_ipv = tf.where(tf.greater_equal(ctr_label, tf.fill(tf.shape(ctr_label), 1.0)),
                                         tf.fill(tf.shape(ctr_label), True),
                                         tf.fill(tf.shape(ctr_label), False))

            with tf.name_scope("cvr_loss"):
                self.cvr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.boolean_mask(cvr_logits, self.mask_ipv),
                    labels=tf.boolean_mask(cvr_label, self.mask_ipv)))

            with tf.name_scope("loss_sum"):
                self.loss_op = self.ctr_loss + 0.5 * self.ctcvr_loss + self.reg_loss \
                               + 0.1 * self.aux_ctr_loss

            return self.loss_op

    def optimizer(self, context, loss_op):
        """
        return train_op
        """
        with self.variable_scope(name_or_scope="Optimize"):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = update_op(name=self.model_name)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = get_optimizer(opt_name, opt_conf, self.global_step)
                learning_rate = SearchAdagrad(opt_conf).get_learning_rate(self.global_step)

                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=learning_rate,
                            optimizer=optimizer,
                            # update_ops=update_ops,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=learning_rate,
                    optimizer=global_optimizer,
                    # update_ops=update_ops,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def predictions(self):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.ctr_prediction = tf.sigmoid(self.ctr_logits)  # (?,1)
            self.cvr_prediction = tf.sigmoid(self.cvr_logits)

    def run_train(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step, self.ctr_loss, self.cvr_loss, self.loss_op,
                       self.metrics, self.ctr_label, self.cvr_label]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, ctr_loss, cvr_loss, loss, metrics, ctr_label, cvr_label = \
                        mon_session.run(run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, ctr_loss, cvr_loss, loss, metrics, ctr_label, cvr_label, _ = \
                        mon_session.run(run_ops, feed_dict=feed_dict)

            except (ResourceExhaustedError, OutOfRangeError) as e:
                break  # release all
            except ConnectionError as e:
                pass
            except Exception as e:
                pass


def update_op(name):
    update_ops = []
    start = 'Share' if name is None else ('Share', name)
    for update_op_ in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        if update_op_.name.startswith(start):
            update_ops.append(update_op_)
    return update_ops


def get_optimizer(opt_name, opt_conf, global_step):
    optimizer = None
    for name in optimizer_dict:
        if opt_name == name:
            optimizer = optimizer_dict[name](opt_conf, global_step)
            break

    return optimizer

