import os

import numpy
import tensorflow as tf

import mes


def get_residual_unit(name, data, num_filter):
    with tf.name_scope(name) as sub_sub_scope:
        bn1 = tf.layers.batch_normalization(inputs=data, epsilon=1e-5)
        conv1 = tf.layers.conv2d(inputs=bn1, filters=num_filter,  kernel_size=(3, 3),
                                 strides=(1, 1), activation=tf.nn.relu,
                                 use_bias=False, padding='same')
        bn2 = tf.layers.batch_normalization(inputs=conv1, epsilon=1e-5)
        conv2 = tf.layers.conv2d(inputs=bn2, filters=num_filter, kernel_size=(3, 3),
                                 strides=(1, 1), activation=tf.nn.relu,
                                 use_bias=False, padding='same')
        return data + conv2


class ValueNet(object):
    def __init__(self, mgame, renew=False, loggable=True):
        self.loggable = loggable
        self.mgame = mgame
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("Input") as scope:
                self.boards = tf.placeholder(tf.float32, shape=[None, mgame.w, mgame.h, 3])

            with tf.name_scope("ResNet") as scope:
                self.conv0 = tf.layers.conv2d(inputs=self.boards, filters=mes.NET_FILTER_NUM, kernel_size=(3, 3),
                                              activation=tf.nn.relu, use_bias=False, padding='same')
                data = self.conv0
                self.res_units = []
                for i in range(mes.NET_LAYER_NUM):
                    res_unit = get_residual_unit("resnet_cell_%d" % (i + 1), data, mes.NET_FILTER_NUM)
                    self.res_units.append(res_unit)
                    data = res_unit

            with tf.name_scope("Actions_Predict") as scope:
                self.p_conv = tf.layers.conv2d(inputs=self.res_units[-1], filters=mes.NET_ACT_LAST_CNN_FILTER_NUM,
                                               kernel_size=(1, 1), activation=tf.nn.relu, use_bias=False,
                                               padding='same')
                self.p_conv_flatten = tf.reshape(self.p_conv,
                                                 [-1, mes.NET_ACT_LAST_CNN_FILTER_NUM * self.mgame.board_sz])
                self.p_dense = tf.layers.dense(inputs=self.p_conv_flatten, units=self.mgame.board_sz,
                                               activation=tf.nn.softmax, name="action_result")

            with tf.name_scope("Evaluation_Predict") as scope:
                self.v_conv = tf.layers.conv2d(inputs=self.res_units[-1], filters=mes.NET_V_LAST_CNN_FILTER_NUM,
                                               kernel_size=(self.mgame.num, self.mgame.num), activation=tf.nn.relu,
                                               use_bias=False,
                                               padding='same')
                self.v_conv_flatten = tf.reshape(self.v_conv, [-1, mes.NET_V_LAST_CNN_FILTER_NUM * self.mgame.board_sz])
                self.v_linear = tf.layers.dense(inputs=self.v_conv_flatten, units=mes.NET_V_LAST_LINEAR_UNIT_NUM,
                                                activation=tf.nn.relu, name="v_linear")
                self.v_dense = tf.layers.dense(inputs=self.v_linear, units=1, activation=tf.nn.tanh, name="v_result")

            with tf.name_scope("Train") as scope:
                with tf.name_scope("Labels") as sub_scope:
                    self.zs = tf.placeholder(tf.float32, shape=[None, 1])
                    self.pies = tf.placeholder(tf.float32, shape=[None, self.mgame.board_sz])
                with tf.name_scope("Loss") as sub_scope:
                    self.value_loss = tf.losses.mean_squared_error(self.zs, self.v_dense)
                    self.p_pie_sum = tf.reduce_sum(
                        tf.multiply(self.pies, tf.log(tf.clip_by_value(self.p_dense, 1e-8, 1.0))), axis=-1)
                    self.policy_loss = tf.negative(tf.reduce_mean(self.p_pie_sum))
                    l2_penalty_beta = 1e-4
                    self.l2_penalty = l2_penalty_beta * tf.add_n(
                        [tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                         if 'bias' not in variable.name.lower()]
                    )
                    self.loss = self.value_loss + self.policy_loss + self.l2_penalty

                self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

            with tf.name_scope("Others") as scope:
                self.saver = tf.train.Saver()
                if self.loggable:
                    tf.summary.scalar('Value loss', self.value_loss)
                    tf.summary.scalar('Policy loss', self.policy_loss)
                    tf.summary.scalar('L2 Penalty', self.l2_penalty)
                    tf.summary.scalar('Loss', self.loss)
                    self.merge_all = tf.summary.merge_all()
                    self.writer = tf.summary.FileWriter(mes.LOG_PATH, self.graph)

            self.session = tf.Session(graph=self.graph)
            if not renew and mes.MODEL_PATH and os.path.exists(mes.MODEL_PATH + ".meta"):
                self.saver.restore(self.session, mes.MODEL_PATH)
                print('Restored from', mes.MODEL_PATH)
            else:
                init = tf.global_variables_initializer()
                self.session.run(init)
        self.data_store = {self.boards: numpy.ndarray([mes.NET_BATCH_LEN, self.mgame.h, self.mgame.w, 3]),
                           self.pies: numpy.ndarray([mes.NET_BATCH_LEN, self.mgame.board_sz]),
                           self.zs: numpy.ndarray([mes.NET_BATCH_LEN, 1])}
        self.data_store_len = 0

    def train_step(self, boards, zs, pies):
        for board, z, pie in zip(boards[2:], zs[2:], pies[2:]):
            self.data_store[self.boards][self.data_store_len] = board
            self.data_store[self.zs][self.data_store_len] = z
            self.data_store[self.pies][self.data_store_len] = pie
            self.data_store_len += 1
            if self.data_store_len == mes.NET_BATCH_LEN:
                if self.loggable:
                    _, summary, loss = self.session.run([self.optimizer, self.merge_all, self.optimizer],
                                                        feed_dict=self.data_store)
                    self.writer.add_summary(summary)
                else:
                    _, loss = self.session.run([self.optimizer, self.optimizer],
                                               feed_dict=self.data_store)

                self.data_store_len = 0

    def predict(self, board):
        feed_dict = {self.boards: numpy.array([board])}
        vs, probs, p_conv_flatten = self.session.run([self.v_dense, self.p_dense, self.p_conv_flatten],
                                                     feed_dict=feed_dict)
        return probs[0], vs[0][0]

    def save(self):
        self.saver.save(self.session, mes.MODEL_PATH)
