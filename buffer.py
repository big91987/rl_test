
class baseBuffer(object):

    def __init__(self, size, gamma):
        self._size = int(size)
        self._gamma = float(gamma)
        # FIFO队列
        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self.value_queue = []
        self._discount = 0.0
        self._len = 0

    def reset(self):
        self.obs_queue = []
        self.reward_queue = []
        self.done_queue = []
        self.value_queue = []
        self._discount = 0.0
        self._len = 0


    # put 的时候自动计算出v_target, adv
    # obs 当前状态
    # 当前状态 采取action以后得到的reward
    # 当前状态 采取action返回的done
    # 当前状态 经过值网络以后的估计值value
    # 注意 起始状态 put(None, s0, None, value[s0])
    #     中间 put(r , s, d, v[s])
    # 结束状态

    def __call__(self, reward, next_obs, done, value):

        if self._len < self._size:
            self.reward_queue.append(reward)
            self.obs_queue.append(next_obs)
            self.done_queue.append(done)
            self.value_queue.append(value)
            self._discount += reward * pow(self._gamma, self._len) * (1 - done)
            self._len += 1
            return
            # return None
        else:
            # 更新discount
            self._discount = (self._discount - self.reward_queue[0]) / self._gamma + \
                             reward * pow(self._gamma, self._len) * (1 - done)
            # 元素前移动
            for i in range(self._size - 1):
                self.obs_queue[i] = self.obs_queue[i + 1]
                self.reward_queue[i] = self.reward_queue[i + 1]
                self.done_queue[i] = self.reward_queue[i + 1]
                self.value_queue[i] = self.value_queue[i + 1]

            self.obs_queue[-1] = next_obs
            self.reward_queue[-1] = reward
            self.done_queue[-1] = done
            self.value_queue[-1] = value

            v_target = self._discount + value
            adv = self.v_target - self.value_queue[0]
            return self.obs_queue[0], v_target, adv