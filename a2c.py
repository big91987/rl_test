import gym
import tensorflow as tf
import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

class AC_network():
    def __init__(self, sess, **params):
        self.sess = sess
        self.params = {}
        self.params.update(params)
        pass

    def load_net(self):
        graph = self.sess.graph
        self.s = graph.get_tensor_by_name('cnn/state:0')
        self.v = graph.get_tensor_by_name('critic/v:0')
        self.acts_prob = graph.get_tensor_by_name('actor/acts_prob:0')

        self.a = graph.get_tensor_by_name('train/a:0')
        self.traget = graph.get_tensor_by_name('train/target:0')
        self.log_prob = graph.get_tensor_by_name('train/log_prob:0')
        self.td_err = graph.get_tensor_by_name('train/td_err:0')
        self.critic_loss = graph.get_tensor_by_name('train/critic_loss:0')
        self.actor_loss = graph.get_tensor_by_name('train/actor_loss:0')

        self.entropy = graph.get_tensor_by_name('train/entropy:0')
        self.loss = graph.get_tensor_by_name('train/loss:0')
        self.train_op = tf.train.AdamOptimizer(
                    0.3).minimize(self.loss)


    def build_net(self,input_shape, action_dim, entropy_c = 0.2):
        #with tf.variable_scope('input')
        with tf.variable_scope('cnn'):
            self.s = tf.placeholder(tf.float32,
                    [1,84,84,4],
                    'state')
            l1 = tf.layers.conv2d(
                inputs=self.s,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding='valid',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            mean=0.0, stddev=5e-2),
                bias_initializer=\
                        tf.constant_initializer(0.0),
                name='l1'
            )

            #l2 = tf.layers.max_pooling2d(
            #    inputs=l1,
            #    pool_size=2,
            #    strides=2,
            #    name='l2'
            #)

            l3 = tf.layers.conv2d(
                inputs=l1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='valid',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            mean=0.0, stddev=5e-2),
                bias_initializer=\
                        tf.constant_initializer(0.0),
                name='l3'
            )

            #l4 = tf.layers.max_pooling2d(
            #    inputs=l3,
            #    pool_size=2,
            #    strides=2,
            #    name='l4'
            #)
            l5 = tf.layers.conv2d(
                inputs=l3,
                filters=64,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='valid',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=\
                        tf.random_normal_initializer(
                            mean=0.0, stddev=5e-2),
                bias_initializer=tf.constant_initializer(0.0),
                name='l5'
            )

            #l6 = tf.layers.max_pooling2d(
            #    inputs=l5,
            #    pool_size=5,
            #    strides=5,
            #    name='l6'
            #)

            l7 = tf.layers.flatten(
                inputs=l5,
                name='l7'
            )

        with tf.variable_scope('critic'):
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
            self.target = tf.placeholder(tf.float32,[1,1], 'target')
            #self.r = tf.placeholder(tf.float32, None, 'r')
            self.log_prob = tf.log(self.acts_prob[0, self.a])
            self.td_err = self.target - self.v
            self.critic_loss = tf.square(self.td_err)
            self.actor_loss = -tf.reduce_mean(self.log_prob * self.td_err, name = 'a_loss')

            self.entropy = \
                    -tf.reduce_sum(
                            self.acts_prob * \
                                tf.log(self.acts_prob),
                            name='entropy')
            self.loss = self.actor_loss + 0.5*self.critic_loss - 0.01* self.entropy
            self.train_op = tf.train.AdamOptimizer(7e-4).minimize(self.loss)
            #self.loss = self.critic_loss - \
            #        self.actor_loss - entropy_c * self.entropy
            #self.train_op = tf.train.AdamOptimizer(
            #        0.1).minimize(self.loss)

    def train_by_td(self, s, r, a, s_n, gamma):
        v_n = self.sess.run(self.v,{self.s:s_n})
        target = r + gamma * v_n
        self.sess.run(self.train_op,
                {self.a:a, self.s:s,self.target:target})

    def train_use_gt(self, ):
        pass

    def choose(self, s):
        a_prob = self.sess.run(self.acts_prob, {self.s:s})
        #a_prob /= np.sqrt(np.sum(np.square(a_prob)))
        #a_prob /= np.sum(a_prob)
        #print('a_prob1 = ' + str(a_prob1))
        #print('a_prob = ' + str(a_prob.ravel()))
        ret = np.random.choice(
                np.arange(a_prob.shape[1]), p=a_prob.ravel())
        return ret

    # export actor pb
    def export_pb(self):
        pass
    def import_pb(self):
        pass

    def export_log(self, log_path = './log/'):
        tf.summary.FileWriter(log_path, self.sess.graph)

    def store(self, save_name ):
        saver = tf.train.Saver()
        saver.save(self.sess, save_name)

    def restore(self, load_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_name)
        pass

def a2c_train(
        log_path = '',
        load_file_name = '',
        save_file_name = '',
        render = False):
    env = gym.make('Breakout-v4')
    env.seed(1)
    #env = env.unwarpped

    N_A = env.action_space.n

    sess = tf.Session()

    ac = AC_network(sess=sess)

    if load_file_name != '':
        ac.restore(load_name=load_file_name)
        ac.load_net()
    else:
        ac.build_net(input_shape = [84,84,4],action_dim=N_A)

    sess.run(tf.global_variables_initializer())

    #img_buffer = []
    img_buffer_size = 4

    max_loop = 100000

    for i in range(max_loop):
        print('loop ' + str(i) + ' begin')
        img_buffer = []
        s = env.reset()
        img_buffer.append(s)

        done = False
        max_k = 3000
        k = 0
        while not done:
            k += 1
            if k > max_k:
                print('wait too long for one test,break')
                break
            if len(img_buffer) < img_buffer_size:
                a = np.random.randint(0, N_A)
                s_, r, done, info = env.step(a)
                img_buffer.append(s_)

                #print ('feed random action = ' + str(a))
            else:
                x = utils.imgbuffer_process(img_buffer)
                x = np.expand_dims(x, 0)

                #plt.subplot(2, 2, 1)
                #plt.imshow(np.uint8(x[0,:, :, 0] * 255), cmap='gray')
                #plt.subplot(2, 2, 2)
                #plt.imshow(np.uint8(x[0,:, :, 1] * 255), cmap='gray')
                #plt.subplot(2, 2, 3)
                #plt.imshow(np.uint8(x[0,:, :, 2] * 255), cmap='gray')
                #plt.subplot(2, 2, 4)
                #plt.imshow(np.uint8(x[0,:, :, 3] * 255), cmap='gray')
                #plt.show()


                a = ac.choose(x)
                #a = np.random.randint(2,6)
                #print ( 'a!! = '+ str(a))
                s_, r, done, info = env.step(a)

                if r > 0 :
                    print('    got a score in loop:' + str(i))
                img_buffer.pop(0)
                img_buffer.append(s_)
                x_ = utils.imgbuffer_process(img_buffer)
                x_ = np.expand_dims(x_,0)
                ac.train_by_td(x,r,a,x_,0.8)

            if render:
                env.render()


        #a = np.random.randint(0, N_A - 1)
        #a = ac.choose()
        #s_, r, done, info = env.step(a)
        #env.render()

        #if len(img_buffer) < img_buffer_size:
        #    img_buffer.append(s_)
        #    continue
        #else:
        #    img_buffer.pop(0)
        #    img_buffer.append(s_)

        #img_input = utils.imgbuffer_process(img_buffer)























    #if log_path != '':
    #    ac.export_log(log_path=log_path)
        #pass
        #tf.summary.FileWriter(log_path, sess.graph)

    #if load_file_name != '':
    #    pass
        #saver = tf.train.Saver()
        #saver.restore(sess, load_file_name)


    #elif mode == 'train':

    #if load_file_name != '':
    #    saver = tf.train.Saver()
    #    saver.save(sess, load_file_name)


def a2c_infer(
        log_path = '',
        pb_file_name = '',
        render = False):
    max_ep = 10
    env = gym.make('Breakout-v4')
    env.seed(1)
    env = env.unwarpped

    sess = tf.Session()

    # make actor net
    ##load_pb
    #sess.run(tf.global_variables_initializer())
    for i_ep in range(max_ep):
        s = env.reset()
        while True:
            if render:env.render()
            #a_prob = actor.infer(s)
           ## choice a_prob
            #s_, r, done, info = env.step(a)
           # if done:
           #     break

def test_a2c_infer():
    a2c_infer(log_path='./log/', pb_name='test.pb',render=True )

def test_a2c_train():
    a2c_train(render=False)



if __name__ == '__main__':
    test_a2c_train()
