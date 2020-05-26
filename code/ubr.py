import tensorflow as tf

class UBRBase(object):
    def __init__(self, feature_size, eb_dim, hidden_size, record_fnum, emb_initializer):
        self.record_fnum = record_fnum

        # input placeholders
        with tf.name_scope('ubr/inputs'):
            self.target_ph = tf.placeholder(tf.int32, [None, record_fnum], name='ubr_target_ph')

            self.rewards = tf.placeholder(tf.float32, [None, 1], name='rewards_ph')
            self.lr = tf.placeholder(tf.float32, [])
        
        # embedding
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            if emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer)
                self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
                self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
                self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

        self.target = tf.nn.embedding_lookup(self.emb_mtx, self.target_ph) #[ B, F, EMB_DIM]
        self.target_input = self.target[:, 1:, :] # exclude uid
    
    def build_index_and_loss(self, probs):
        uniform = tf.random_uniform(tf.shape(probs), 0, 1)
        condition = probs - uniform
        self.index = tf.where(condition >= 0, tf.ones_like(probs), tf.zeros_like(probs))
        log_probs = tf.log(tf.clip_by_value(probs, 1e-10, 1))

        self.loss = -tf.reduce_mean(tf.reduce_sum(log_probs * self.index * self.rewards, axis=1))
        self.reward = tf.reduce_mean(self.rewards)
        tf.summary.scalar('ubr_reward', self.reward)
        self.merged = tf.summary.merge_all()

    def build_optimizer(self):    
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='ubr_adam')
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = []
        for grad, var in gvs:
            if grad is not None:
                capped_gvs.append((tf.clip_by_norm(grad, 5.), var))
        self.train_step = self.optimizer.apply_gradients(capped_gvs)
        # self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, rewards):
        loss, reward, _, summary = sess.run([self.loss, self.reward, self.train_step, self.merged], feed_dict = {
            self.target_ph : batch_data,
            self.lr : lr,
            self.rewards : rewards
        })
        return loss, reward, summary

    def get_distri(self, sess, batch_data):
        res = sess.run(self.probs, feed_dict={
            self.target_ph : batch_data
        })
        return res
    
    def get_index(self, sess, batch_data):
        res = sess.run(self.index, feed_dict={
            self.target_ph : batch_data
        })
        return res

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))


class UBR_SA(UBRBase):
    def __init__(self, feature_size, eb_dim, hidden_size, record_fnum, emb_initializer):
        super(UBR_SA, self).__init__(feature_size, eb_dim, hidden_size, record_fnum, emb_initializer)
        self.probs = self.build_select_probs(self.target_input)
        self.build_index_and_loss(self.probs)
        self.build_optimizer()
        
    def build_select_probs(self, target_input):
        sa_target = self.multihead_attention(self.normalize(target_input), target_input)
        probs = tf.layers.dense(sa_target, 20, activation=tf.nn.relu, name='fc1')
        probs = tf.layers.dense(probs, 10, activation=tf.nn.relu, name='fc2')
        probs = tf.layers.dense(probs, 1, activation=tf.nn.sigmoid, name='fc3')
        probs = tf.reshape(probs, [-1, self.record_fnum - 1])
        return probs


    def multihead_attention(self,
                            queries, 
                            keys, 
                            num_units=None, 
                            num_heads=2, 
                            scope="multihead_attention", 
                            reuse=None):
        '''Applies multihead attention.
        
        Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
        A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
            
            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
    
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
            
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
            
            # Dropouts
            outputs = tf.nn.dropout(outputs, 0.8)
                
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                
            # Residual connection
            outputs += queries
                
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
    
        return outputs

    def normalize(self,
              inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
        '''Applies layer normalization.
        
        Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
        Returns:
        A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs
