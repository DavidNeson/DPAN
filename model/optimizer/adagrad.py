import tensorflow as tf
import math


class SearchAdagrad:
    def __init__(self, conf):
        self.conf = conf

    def get_optimizer(self, global_step):
        learning_rate = self.get_learning_rate(global_step)
        tf.summary.scalar(name="Optimize/learning_rate", tensor=learning_rate)

        return tf.train.AdagradOptimizer(learning_rate)

    def get_learning_rate(self, global_step):
        init_learning_rate = self.conf['learning_rate']
        # min_learning_rate = self.conf['min_learning_rate']
        # decay_step = self.conf['decay_step']
        # learning_rate = tf.where(global_step < decay_step, init_learning_rate, min_learning_rate)
        learning_rate = init_learning_rate

        return learning_rate
