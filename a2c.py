


class AC_network():
    def __init__(self):
        pass

    def build_net(self):

def a2c_train(
        log_path = '',
        load_file_name = '',
        render = False):
    env = gym.make('Breakout-v4')
    env.seed(1)
    env = env.unwarpped

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    if log_path != '':
        tf.summary.FileWriter(log_path, sess.graph)

    if load_file_name != '':
        saver = tf.train.Saver()
        saver.restore(sess, load_file_name)

    if mode == 'infer':
        for i_ep in range(MAX_EP):
            s = env.reset()

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
