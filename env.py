#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Dagui Chen
Email: goblin_chen@163.com

``````````````````````````````````````
Env Class

Some Wrapper class to make environment trainable

Modify from:
    https://github.com/openai/baselines
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from buffer import baseBuffer as Buffer

import cv2
import gym
from gym import spaces
import numpy as np
from multiprocessing import Process, Pipe
from collections import deque
import os
import matplotlib
import time
from buffer import baseBuffer as Buffer

# 初始env加上若干步空操作（noop）
class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking randonm number of no-ops of reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30, noop_action=0):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = noop_action
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

#
class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing
    """
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

# 每隔若干帧返回一次
class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame
    """
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


# 剪裁reward
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign
        """
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over/
    Done by Deepmind for the DQN and co. since it helps value estimation
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0  # for some game, this means number of lives
        self.was_real_done = True

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info


# 将图像resize，并灰化
# 观测装饰器
class WrapFrame(gym.ObservationWrapper):
    """Wrap frames to 84x84 as done in the Nature paper and later work
    """
    def __init__(self, env, width=84, height=84):
        super(WrapFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


# 将灰度图图片累计n张转换成tensor
class StackFrame(gym.Wrapper):
    """Stack the frames as the state
    """
    def __init__(self, env, num_stack):
        super(StackFrame, self).__init__(env)
        self.num_stack = num_stack
        self.obs_queue = deque(maxlen=self.num_stack)
        raw_os = self.env.observation_space
        low = np.repeat(raw_os.low, num_stack, axis=-1)
        high = np.repeat(raw_os.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=raw_os.dtype)

    def reset(self):
        obs = self.env.reset()
        for i in range(self.num_stack - 1):
            self.obs_queue.append(np.zeros_like(obs, dtype=np.uint8))
        self.obs_queue.append(obs)
        stack_obs = np.concatenate(self.obs_queue, axis=-1)  # [..., c * num_stack]
        return stack_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_queue.append(obs)
        stack_obs = np.concatenate(self.obs_queue, axis=-1)
        return stack_obs, reward, done, info


class Worker(object):
    def __init__(self, **kwargs):
        assert 'worker_port' in kwargs.keys(), 'worker_port needed... worker_proc quit'
        assert 'env_id' in kwargs.keys(), 'env_id needed... worker_proc quit'
        assert 'seed' in kwargs.keys(), 'seed need ... slave_proc quit'
        # assert 'slave_pipe' in kwargs.keys(), 'need slave pipe ... slave_proc quit'
        assert 'gamma' in kwargs.keys(), 'gamma is needed for buffer'
        assert 'buffer_size' in kwargs.keys(), 'buffer_size is needed for buffer'

        self.producer = Buffer(gamma=kwargs['gamma'], size=kwargs['buffer_size'])

        self._debug = True if ('debug' in kwargs.keys() and kwargs['debug'] is True) else False
        self._render = True if ('render' in kwargs.keys() and kwargs['render'] is True) else False
        self._worker_id = kwargs['worker_id'] if 'worker_id' in kwargs.keys() else os.getpid()

        self._log(info='[env_init]env_id = {}'.format(kwargs['env_id']))
        self._env = gym.make(kwargs['env_id'])

        if 'noop_max' in kwargs.keys():
            assert isinstance(kwargs['noop_max'], int)
            self._log(info='[env_init]wrapper noop_max {}'.format(kwargs['noop_max']))
            self._env = NoopResetEnv(self._env, noop_max=kwargs['noop_max'])

        if 'skip' in kwargs.keys():
            self._log(info='[env_init]wrapper skip {}'.format(kwargs['skip']))
            assert isinstance(kwargs['skip'], int)
            self._env = MaxAndSkipEnv(self._env, skip=kwargs['skip'])

        if 'episodic_life' in kwargs.keys():
            self._log(info='[env_init]wrapper episodicLifeEnv')
            self._env = EpisodicLifeEnv(self._env)

        if 'FIRE' in self._env.unwrapped.get_action_meanings():
            self._log(info='[env_init]wrapper FireResetEnv')
            self._env = FireResetEnv(self._env)

        self._log(info='[env_init]wrapper WrapFrame')
        self._env = WrapFrame(self._env)

        if 'clip_reward' in kwargs.keys():
            self._log(info='[env_init]wrapper ClipRewardEnv')
            self._env = ClipRewardEnv(self._env)

        self._log(info='[env_init]seed = {}'.format(kwargs['seed']))
        self._env.seed(kwargs['seed'])

        if 'num_stack' in kwargs.keys():
            self._log(info='[env_init]wrapper stack {} frames'.format(kwargs['num_stack']))
            self._env = StackFrame(self._env, num_stack=kwargs['num_stack'])

        self._port = kwargs['worker_port']

        self.status = {
            'obs': None,
            'done': None,
        }

    def _log(self, info):
        if not self._debug:
            return
        print('[worker: {worker_id}]{info}'.format(worker_id=self._worker_id, info=info))

    def listen(self):
        n_step = 0

        while True:
            cmd, data = self._port.recv()
            if cmd == "step":
                n_step += 1
                self._log(info='[step {}]action = {}'.format(n_step, data))
                obs, reward, done, info = self._env.step(data[:-2])
                self._log(info='[step {}]reward = {}'.format(n_step, reward))
                # if done:
                #     self._log(info='done ... reset'.format(n_step))
                #     obs = self._env.reset()
                train_tuple = self.producer(done=done,reward=reward,next_obs=obs,value=data[-1])
                self._port.send((obs, reward, done, info, train_tuple))
                self.status['obs'] = obs
                self.status['done'] = done

                if self._render:
                    self._env.render()
            if cmd == "fit":
                n_step += 1
                action, value = data
                self._log(info='[step {}]action, value = {}'.format(n_step, data))
                obs, reward, done, info = self._env.step(action=action)
                self._log(info='[step {}]reward = {}'.format(n_step, reward))
                # if done:
                #     self._log(info='done ... reset'.format(n_step))
                #     obs = self._env.reset()
                train_tuple = self.producer(done=done,reward=reward,next_obs=obs,value=value)
                self._port.send(train_tuple)
                self.status['obs'] = obs
                self.status['done'] = done

                if self._render:
                    self._env.render()
            elif cmd == "reset":
                self._log(info='reset')
                n_step = 0
                obs = self._env.reset()
                self._port.send(obs)
            elif cmd == "shutdown":
                self._log(info='shutdown ...')
                self._port.send('done')
                break
            elif cmd == "get_info":
                self._log(info='get env info ...')
                data = (self._env.observation_space, self._env.action_space)
                self._port.send(data)
            elif cmd == 'get_status':
                self._log(info='get env status ...')
                data = self.status
                self._port.send(data)
            else:
                raise NotImplementedError


class pEnv(object):

    @classmethod
    def work(cls, **kwargs):
        worker = Worker(**kwargs)
        worker.listen()

    def __init__(self, **kwargs):
        assert 'n_env' in kwargs.keys(), 'n_env need for setup env ...'
        assert 'env_id' in kwargs.keys(), 'env_id needed for setup env ...'
        self._n_env = kwargs['n_env']

        assert self._n_env > 0, 'n_env must > 0 ...'
        kwargs.pop('n_env')

        self.master_ports = []
        self.close = False
        self.seed = kwargs['seed'] if 'seed' in kwargs.keys() else 0

        for i in range(self._n_env):

            master_port, worker_port = Pipe()
            self.master_ports.append(master_port)
            if 'seed' in kwargs.keys():
                kwargs.pop('seed')
            if 'worker_port' in kwargs.keys():
                kwargs.pop('worker_port')
            kwargs['seed'] = self.seed + i
            kwargs['worker_port'] = worker_port
            kwargs['worker_id'] = i
            p = Process(target=self.work, kwargs=kwargs)
            p.daemon = True
            p.start()
            worker_port.close()

        self.batch_buffer = []

    def reset(self):
        for master_port in self.master_ports:
            master_port.send(("reset", None))
        results = [master_port.recv() for master_port in self.master_ports]
        return results

    def shutdown(self):
        for master_port in self.master_ports:
            master_port.send(("shutdown", None))
        # results = [master_port.recv() for master_port in self.master_ports]
        # return results

    # def step(self, actions):
    #     for master_port, action in zip(self.master_ports, actions):
    #         master_port.send(('step', action))
    #     results = zip(*[master_port.recv() for master_port in self.master_ports])
    #     return results

    def fit(self, actions, values):
        for master_port, action, value in zip(self.master_ports, actions, values):
            master_port.send(('fit', (action, value)))
        # results = zip(*filter(lambda x: x is not None, [master_port.recv() for master_port in self.master_ports]))
        # return results
        tmp =  filter(lambda x: x is not None, [master_port.recv() for master_port in self.master_ports])
        # 传入到buffer中
        for t in tmp:
            self.batch_buffer.append(t)

    def get_batch(self, batch_size):
        if len(self.batch_buffer) < batch_size:
            return None
        else:
            tmp = self.batch_buffer[:batch_size-1]
            self.batch_buffer = self.batch_buffer[batch_size:]
            return zip(*tmp)

    def get_env_info(self):
        self.master_ports[0].send(('get_info', None))
        ret = [self.master_ports[0].recv()]
        return ret

    # def is_all_done(self):
    #     for master_port in self.master_ports:
    #         master_port.send(('step'))
    #     results = zip(*[master_port.recv() for master_port in self.master_ports])
    #     return results

    def render(self, mode, **kwargs):
        raise NotImplementedError

    def unwrapped(self):
        raise NotImplementedError


def test_pEnv():
    n_env = 32

    env = pEnv(env_id='BreakoutNoFrameskip-v4',
               n_env=n_env,
               debug=False,
               render=False,
               num_stack=4)

    print('info of env0 {}'.format(list(env.get_env_info())))

    env.reset()

    for _ in range(100000):
        a = []
        for i in range(n_env):
            a.append(np.random.randint(0,4))
        results = env.step(actions=a)

    env.shutdown()

if __name__ == '__main__':
    test_pEnv()
