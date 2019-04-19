import tensorflow.keras as keras

import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

# class CNNModel(object):
#     def __init__(self):
#         pass
#     def __call__(self, *args, **kwargs):
#         # if not 'input' in kwargs.keys():
#         #     print('input need')
#         #     return
#         # input = kwargs['input']
#
#         model = keras.Sequential()
#         model.add(keras.layers.Conv2D(
#             filters=32,
#             kernel_size=8,
#             strides=(3, 3),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Conv2D(
#             filters=64,
#             kernel_size=4,
#             strides=(2, 2),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Conv2D(
#             filters=64,
#             kernel_size=8,
#             strides=(1, 1),
#             padding='valid',
#             kernel_initializer='he_normal',
#         ))
#         model.add(keras.layers.Activation('relu'))
#         model.add(keras.layers.Flatten())


class Td_n_Buffer(object):

    def __init__(self,buffer_size, gamma, worker_num = 1, **kwargs):
        self.buffer_size = buffer_size
        self.gamma = gamma
        # self.worker_num = worker_num
        # FIFO队列
        # 注意rl是时间相关序列问题，每个worker的轨迹具有时间相关性不能shuffle，不同worker的结果并行存储到FIFObuffer中
        self.reset()

        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self._len = 0

    def reset(self):
        # self.obs_queue = [None] * self.buffer_size
        # self.reward_queue = [None] * self.buffer_size
        # self.done_queue = [None] * self.buffer_size
        # self.value_queue = [None] * self.buffer_size
        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self._len = 0




        # self.action_queue = [None] * self.buffer_size
        # self.action_prob_queue = [None] * self.buffer_size
        # self.info

        # self.adv_queue = [None] * self.buffer_size
        # self.gt_queue = [None] * self.buffer_size
        # self.buffer_num = 0
        # self.step_num = 0

    # # 获取k步折扣
    # def get_discount(self, k):
    #     assert k <= self.buffer_size
    #     # r0, r1*gamma, r2*gam^2 ... rn-1*gam^n-1
    #     dis_list = [self.reward_queue[i] * np.power(self.gamma, float(i)) for i in range(len(self.reward_queue))]
    #
    #     coef = np.power(self.gamma, range(len(self.reward_queue)))
    #     return sum(dis_list)

    def get_target_and_obs(self, v_future = None):
        # coef = np.power(self.gamma, range(len(self.value_queue-1)))
        # return np.sum(coef * self.reward_queue[:-2]) + self.value_queue[-1]
        # return get_
        tmp = np.zeros_like(self.obs_queue[0])
        len = len(self.obs_queue)
        for i in range(len):
            # tmp = self.gamma * (self.reward_queue[len - i] * (1 - self.done_queue[len - i]) + tmp)
            tmp += pow(self.gamma, i) * self.reward_queue[i] * (1 - self.done_queue[i])

        return tmp + v_future, self.obs_queue[0]


    # def put(self, reward, obs, done, value):
    #     for i in reversed(range(1, self.buffer_size)):
    #         self.obs_queue[i] = self.obs_queue[i-1]
    #         self.reward_queue[i] = self.reward_queue[i-1]
    #         self.done_queue[i] = self.reward_queue[i-1]
    #         self.value_queue[i] = self.value_queue[i-1]
    #     self.obs_queue[0] = obs
    #     self.reward_queue[0] = reward
    #     self.done_queue[0] = done
    #     self.value_queue[0] = value
    #     self.buffer_num = self.buffer_num + 1 if self.buffer_num < self.buffer_size else self.buffer_size
    #     self.step_num = self.step_num + 1

    def put(self, reward, obs, done):

        if self._len < self.buffer_size:
            self.reward_queue.append(reward)
            self.obs_queue.append(obs)
            self.done_queue.append(done)
        else:

            for i in range(self.buffer_size - 1):
                self.obs_queue[i] = self.obs_queue[i+1]
                self.reward_queue[i] = self.reward_queue[i+1]
                self.done_queue[i] = self.reward_queue[i+1]

            self.obs_queue[-1] = obs
            self.reward_queue[-1] = reward
            self.done_queue[-1] = done

        self._len += 1
        # self.buffer_num = self.buffer_num + 1 if self.buffer_num < self.buffer_size else self.buffer_size
        # self.step_num = self.step_num + 1

    # # 获取n步td
    # def get_adv(self):
    #
    #     if self.buffer_num < self.buffer_size:
    #         return np.zeros_like(self.reward_queue[0])
    #
    #     discount = self.reward_queue[self.buffer_num - 1]
    #     v = self.value_queue[self.buffer_num - 1]
    #     for t in reversed(range(self.buffer_num)):
    #         discount = self.reward_queue[t] + self.gamma * discount * (1.0-self.done_queue[t])
    #         v = v * self.gamma
    #
    #     return discount + v - self.value_queue[0]
    #
    #
    #     # 获取n步td


class Buffer(object):

    def __init__(self, buffer_size, gamma):
        self.buffer_size = buffer_size
        self.gamma = gamma
        # FIFO队列
        self.reset()

        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self._len = 0

    def reset(self):
        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self._len = 0

    # 但会v的估计值，以及obs，用于训练    v_target + v_future  ===> 过网络的 v(obs)
    def get_target_and_obs(self, v_future=None):
        tmp = np.zeros_like(self.obs_queue[0])
        for i in range(len(self.obs_queue)):
            tmp += pow(self.gamma, i) * self.reward_queue[i] * (1 - self.done_queue[i])

        return tmp + v_future, self.obs_queue[0]

    def put(self, reward, obs, done):

        if self._len < self.buffer_size:
            self.reward_queue.append(reward)
            self.obs_queue.append(obs)
            self.done_queue.append(done)
        else:

            for i in range(self.buffer_size - 1):
                self.obs_queue[i] = self.obs_queue[i + 1]
                self.reward_queue[i] = self.reward_queue[i + 1]
                self.done_queue[i] = self.reward_queue[i + 1]

            self.obs_queue[-1] = obs
            self.reward_queue[-1] = reward
            self.done_queue[-1] = done

        self._len += 1



def cnn_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=(3, 3),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=8,
        strides=(1, 1),
        padding='valid',
        kernel_initializer='he_normal',
    ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Flatten())
    return model

def model_generator(model_name, model_path = None):
    if model_name == 'cnn_modle':
        return cnn_model()
    else:
        return None

from model import model_generator


# 从分布中采样，输入时离散的分布，采样返回index
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=-1)


class A2CModel(tf.keras.Model):
    def __init__(self, num_actions, base_model=None):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.base_model = base_model
        # self.hidden1 = kl.Dense(128, activation='relu')
        # self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    # 返回预测出的logit以及value
    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        # hidden_logs = self.hidden1(x)
        # hidden_vals = self.hidden2(x)
        base = self.base_model(x)
        return self.logits(base), self.value(base)

    # 返回根据预测logit采样出的结果以及value
    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                # 记录episode的奖励
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews) - 1, ep_rews[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            # 两个y是用来fit self._logits_loss, self._value_loss的
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
        return ep_rews

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        # bootstrap 自举 用未来估计现在
        returns = np.append(arr=np.zeros_like(rewards), values=next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        # 求轨迹每个t的Gt，即当前步到最后一步的Gt
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        # adv = Gt - V
        advantages = returns - values
        return returns, advantages

    # 计算值函数损失
    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, value)

    # 计算策略梯度
    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        # Advantage actor-critic形式的策略梯度
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        # 添加策略的熵，增加随机性
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy'] * entropy_loss


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    model = A2CModel(num_actions=env.action_space.n)
    agent = A2CAgent(model)

    rewards_history = agent.train(env)
    print("Finished training.")
    print("Total Episode Reward: %d out of 200" % agent.test(env, True))

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()