import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
import numpy as np

class RecBase(object):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        # input placeholders
        with tf.name_scope('rec/inputs'):
            self.seq_ph = tf.placeholder(tf.int32, [None, b_num, record_fnum], name='seq_ph')
            self.seq_length_ph = tf.placeholder(tf.int32, [None,], name='seq_length_ph')
            self.target_ph = tf.placeholder(tf.int32, [None, record_fnum], name='target_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            if emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=emb_initializer, )
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
                self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
                self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
                self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

        self.seq = tf.nn.embedding_lookup(self.emb_mtx, self.seq_ph)
        self.seq = tf.reshape(self.seq, [-1, b_num, record_fnum * eb_dim])
        self.target = tf.nn.embedding_lookup(self.emb_mtx, self.target_ph)
        self.target = tf.reshape(self.target, [-1, record_fnum * eb_dim])


    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='rec_bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='rec_fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='rec_dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='rec_fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='rec_dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='rec_fc3')
        score = tf.nn.softmax(fc3)
        # output
        self.y_pred = tf.reshape(score[:,0], [-1,])
    
    def build_reward(self):
        # rig as reward (reward)
        self.ground_truth = tf.cast(self.label_ph, tf.float32)
        self.reward = self.ground_truth * tf.log(tf.clip_by_value(self.y_pred, 1e-10, 1)) + (1 - self.ground_truth) * tf.log(1 - tf.clip_by_value(self.y_pred, 1e-10, 1))
        self.reward = 1 - (self.reward / tf.log(0.5)) # use RIG as reward signal
        self.edge = -tf.ones_like(self.reward)
        self.reward = tf.where(self.reward < -1, self.edge, self.reward)

    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
    
    def build_optimizer(self):    
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='rec_optimizer')
        self.train_step = self.optimizer.minimize(self.loss)


    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.seq_ph : batch_data[0],
                self.seq_length_ph : batch_data[1],
                self.target_ph : batch_data[2],
                self.label_ph : batch_data[3],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict = {
                self.seq_ph : batch_data[0],
                self.seq_length_ph : batch_data[1],
                self.target_ph : batch_data[2],
                self.label_ph : batch_data[3],
                self.reg_lambda : reg_lambda,
                self.keep_prob : 1.
            })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss
    
    def get_reward(self, sess, batch_data):
        reward = sess.run(self.reward, feed_dict = {
            self.seq_ph : batch_data[0],
            self.seq_length_ph : batch_data[1],
            self.target_ph : batch_data[2],
            self.label_ph : batch_data[3],
            self.keep_prob : 1.
        })
        return np.reshape(reward, [-1, 1]) #[B,1]

    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))

class RecSum(RecBase):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        super(RecSum, self).__init__(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)

        # use sum pooling to model the user behaviors, padding is zero (embedding id is also zero)
        user_behavior_rep = tf.reduce_sum(self.seq, axis=1)
        
        inp = tf.concat([user_behavior_rep, self.target], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_reward()
        self.build_logloss()
        self.build_optimizer()

class RecAtt(RecBase):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        super(RecAtt, self).__init__(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
        mask = tf.sequence_mask(self.seq_length_ph, b_num, dtype=tf.float32)
        self.atten, user_behavior_rep = self.attention(self.seq, self.seq, self.target, mask)
        self.atten = tf.reshape(self.atten, [-1, b_num])
        inp = tf.concat([user_behavior_rep, self.target], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_reward()
        self.build_logloss()
        self.build_optimizer()


    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1]) # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)
        
        mask = tf.equal(mask, tf.ones_like(mask)) #[B, T]
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.nn.softmax(tf.where(mask, atten, paddings)) #[B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res

