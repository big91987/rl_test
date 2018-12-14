import gym
import tensorflow as tf
import matplotlib as plt


class AC_network():
    def __init__(self):
        self.sess = tf.Session()
        pass

    def build_net(self,state_shape, action_dim):
        #with tf.variable_scope('input')
        with tf.variable_scope('cnn'):
            self.s = tf.placeholder(tf.float32,
                    [1,
                        state_shape[0],
                        state_shape[0],state_shape[0]],
                    'state')
            l1 = tf.layers.conv2d(
                inputs=self.s,
                filters=1,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            mean=0.0, stddev=5e-2),
                bias_initializer=\
                        tf.constant_initializer(0.0),
                name='l1'
            )

            l2 = tf.layers.max_pooling2d(
                inputs=l1,
                pool_size=2,
                strides=2,
                name='l2'
            )

            l3 = tf.layers.conv2d(
                inputs=l2,
                filters=1,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializerz(
                            mean=0.0, stddev=5e-2),
                bias_initializer=\
                        tf.constant_initializer(0.0),
                name='l3'
            )

            l4 = tf.layers.max_pooling2d(
                inputs=l3,
                pool_size=2,
                strides=2,
                name='l4'
            )
            l5 = tf.layers.conv2d(
                inputs=l4,
                filters=1,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            mean=0.0, stddev=5e-2),
                bias_initializer=tf.constant_initializer(0.0),
                name='l5'
            )

            l6 = tf.layers.max_pooling2d(
                inputs=l5,
                pool_size=5,
                strides=5,
                name='l6'
            )

            l7 = tf.layers.flatten(
                inputs=l6,
                name='l7'
            )

        with tf.variable_scope('critic')
            self.v = tf.layers.dense(
                inputs=l7,
                units=1,  # output units
                activation=None,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            0., 5e-2),  # weights
                bias_initializer=tf.constant_initializer(
                    0.0),  # biases
                name='v'
            )

        with tf.variable_scope('actor'):
            self.acts_prob = tf.layers.dense(
                    inputs=l7,
                    units= action_dim,
                    activation= tf.nn.softmax,
                    kernel_initializer= \
                            tf.random_normal_initializer(
                                mean=0.0,stddev=5e-2),
                    bias_initializer= tf.constant_initializer(0.0),
                    name= 'acts_prob'
                )
        with tf.variable_scope('train'):
            self.a = tf.placeholder(tf.int32, None, "action")
            # target can be Gt or r+v_next or Q or ..
            self.traget = tf.placeholder(tf.float32,[1,1], 'target')
            #self.r = tf.placeholder(tf.float32, None, 'r')
            self.log_prob = tf.log(self.actor_loss[0, self.a])
            self.td_err = self.target - self.v
            self.critic_loss = tf.square(self.td_err)
            self.actor_loss = tf.reduce_mean(
                    self.log_prob * self.td_err, name = 'exp_v')

            self.entropy = \
                    -tf.reduce_sum(
                            self.acts_prob * \
                                tf.log(self.acts_prob),
                            name='entropy')
            self.loss = self.critic_loss - \
                    self.actor_loss - entropy_c * self.entropy
            self.train_op = tf.train.AdamOptimizer(
                    self.params['LR']).minimize(self.loss)

    def train_by_td(self, s, r, a, s_n, gamma):
        v_n = self.sess.run(self.v,{self.s:s_n})
        target = r + gamma * v_n
        self.sess.run(self.train_op,
                {self.a:a, self.s:s,self.target:target})

    def train_use_gt(self, ):
        pass

    def choose(s):
        a_prob = self.sess.run(self.acts_prob, {self.s:s})
        a_prob /= np.sqrt(np.sum(np.square(a_prob)))
        a_prob /= np.sum(a_prob)
        #print('a_prob1 = ' + str(a_prob1))
        # print('a_prob = ' + str(a_prob.ravel()))
        ret = np.random.choice(
                np.arange(a_prob.shape[1]), p=a_prob.ravel())
        return ret

    # export actor pb
    def export_pb(self):
        pass
    def import_pb(self):
        pass

    def export_log(self, log_path = './log/'):
        tf.summary.FileWriter(log_path, sess.graph)

    def store(self, save_name ):
        saver = tf.train.Saver()
        saver.save(sess, save_name)

    def restore(self, load_name):
        saver = tf.train.Saver()
        saver.restore(sess, load_name)
        pass


def a2c_train(
        log_path = '',
        load_file_name = '',
        render = False):
    env = gym.make('Breakout-v4')
    env.seed(1)
    env = env.unwarpped

    sess = tf.Session()

    ac = AC_network()
    ac.build_net(state_shape= env.observation_space.shape, action_dim= env.action_space.n)

    sess.run(tf.global_variables_initializer())

    if log_path != '':
        pass
        #tf.summary.FileWriter(log_path, sess.graph)

    if load_file_name != '':
        pass
        #saver = tf.train.Saver()
        #saver.restore(sess, load_file_name)


    #elif mode == 'train':







    if load_file_name != '':
        saver = tf.train.Saver()
        saver.save(sess, load_file_name)


def a2c_infer(
        log_path = '',
        pb_file_name = '',
        render = False):
    env = gym.make('Breakout-v4')
    env.seed(1)
    env = env.unwarpped

    sess = tf.Session()

    # make actor net
    load_pb
    #sess.run(tf.global_variables_initializer())
    for i_ep in range(max_ep):
        s = env.reset()
        while True:
            if render:env.render()
            a_prob = actor.infer(s)
            choice a_prob
            s_, r, done, info = env.step(a)
            if done:
                break

def test_a2c_infer():
    a2c_infer(log_path='./log/', pb_name='test.pb',render=True )



if __name__ == '__main__':
    test_a2c_infer()
