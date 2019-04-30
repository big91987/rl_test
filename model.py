import tensorflow.keras as keras

import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko




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


# class A2CAgent:
#     def __init__(self, model):
#         # hyperparameters for loss terms, gamma is the discount coefficient
#         self.params = {
#             'gamma': 0.99,
#             'value': 0.5,
#             'entropy': 0.0001
#         }
#         self.model = model
#         self.model.compile(
#             optimizer=ko.RMSprop(lr=0.0007),
#             # define separate losses for policy logits and value estimate
#             loss=[self._logits_loss, self._value_loss]
#         )
#
#     def train(self, env, batch_sz=32, updates=1000):
#         # storage helpers for a single batch of data
#         actions = np.empty((batch_sz,), dtype=np.int32)
#         rewards, dones, values = np.empty((3, batch_sz))
#         observations = np.empty((batch_sz,) + env.observation_space.shape)
#         # training loop: collect samples, send to optimizer, repeat updates times
#         ep_rews = [0.0]
#         next_obs = env.reset()
#         for update in range(updates):
#             for step in range(batch_sz):
#                 observations[step] = next_obs.copy()
#                 actions[step], values[step] = self.model.action_value(next_obs[None, :])
#                 next_obs, rewards[step], dones[step], _ = env.step(actions[step])
#                 # 记录episode的奖励
#                 ep_rews[-1] += rewards[step]
#                 if dones[step]:
#                     ep_rews.append(0.0)
#                     next_obs = env.reset()
#                     logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews) - 1, ep_rews[-2]))
#
#             _, next_value = self.model.action_value(next_obs[None, :])
#             returns, advs = self._returns_advantages(rewards, dones, values, next_value)
#             # a trick to input actions and advantages through same API
#             acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
#             # performs a full training step on the collected batch
#             # note: no need to mess around with gradients, Keras API handles it
#             # 两个y是用来fit self._logits_loss, self._value_loss的
#             losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
#             logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
#         return ep_rews
#
#     def test(self, env, render=False):
#         obs, done, ep_reward = env.reset(), False, 0
#         while not done:
#             action, _ = self.model.action_value(obs[None, :])
#             obs, reward, done, _ = env.step(action)
#             ep_reward += reward
#             if render:
#                 env.render()
#         return ep_reward
#
#     def _returns_advantages(self, rewards, dones, values, next_value):
#         # next_value is the bootstrap value estimate of a future state (the critic)
#         # bootstrap 自举 用未来估计现在
#         returns = np.append(arr=np.zeros_like(rewards), values=next_value, axis=-1)
#         # returns are calculated as discounted sum of future rewards
#         # 求轨迹每个t的Gt，即当前步到最后一步的Gt
#         for t in reversed(range(rewards.shape[0])):
#             returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
#         returns = returns[:-1]
#         # advantages are returns - baseline, value estimates in our case
#         # adv = Gt - V
#         advantages = returns - values
#         return returns, advantages
#
#     # 计算值函数损失
#     def _value_loss(self, returns, value):
#         # value loss is typically MSE between value estimates and returns
#         return self.params['value'] * kls.mean_squared_error(returns, value)
#
#     # 计算策略梯度
#     def _logits_loss(self, acts_and_advs, logits):
#         # a trick to input actions and advantages through same API
#         actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
#         # sparse categorical CE loss obj that supports sample_weight arg on call()
#         # from_logits argument ensures transformation into normalized probabilities
#         weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
#         # policy loss is defined by policy gradients, weighted by advantages
#         # note: we only calculate the loss on the actions we've actually taken
#         actions = tf.cast(actions, tf.int32)
#         # Advantage actor-critic形式的策略梯度
#         policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
#         # entropy loss can be calculated via CE over itself
#         # 添加策略的熵，增加随机性
#         entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
#         # here signs are flipped because optimizer minimizes
#         return policy_loss - self.params['entropy'] * entropy_loss
#

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

    def train(self, env, batch_sz=10, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()







        # for update in range(updates):
        #     for step in range(batch_sz):
        #         observations[step] = next_obs.copy()
        #         actions[step], values[step] = self.model.action_value(next_obs[None, :])
        #         next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        #         # 记录episode的奖励
        #         ep_rews[-1] += rewards[step]
        #         if dones[step]:
        #             ep_rews.append(0.0)
        #             next_obs = env.reset()
        #             logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews) - 1, ep_rews[-2]))
        #
        #     _, next_value = self.model.action_value(next_obs[None, :])
        #     returns, advs = self._returns_advantages(rewards, dones, values, next_value)
        #     # a trick to input actions and advantages through same API
        #     acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        #     # performs a full training step on the collected batch
        #     # note: no need to mess around with gradients, Keras API handles it
        #     # 两个y是用来fit self._logits_loss, self._value_loss的
        #     losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        #     logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
        # return ep_rews

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