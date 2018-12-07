import numpy as np
import tensorflow as tf
import copy as cp
import time
import tkinter as tk

class env2048:
    def __init__(self):
        self.W = 4
        self.H = 4
        self.action_set = ['u','d','l','r']
        #self.state = np.zeros((self.W, self.H))
        self.reset()

    def reset(self):
        self.state = self.state = np.zeros((self.W, self.H))
        x0 = np.random.randint(0, self.W-1)
        y0 = np.random.randint(0, self.H-1)
        self.state[x0][y0] = pow(2, np.random.randint(1,2))

        while True:
            x1 = np.random.randint(0, self.W - 1)
            y1 = np.random.randint(0, self.H - 1)
            if self.state[x1][y1] == 0:
                self.state[x0][y0] = pow(2, np.random.randint(1,2))
                break

    def check_done(self):
        for i in range(self.H):
            for j in range(self.W):
                u,d,l,r = -1,-1,-1,-1
                if not i <= 0 :
                    u = self.state[i-1][j]
                if not j <= 0 :
                    l = self.state[i][j-1]
                if not i >= self.H - 1:
                    d = self.state[i+1][j]
                if not j >= self.W - 1:
                    r = self.state[i][j+1]
                o = self.state[i][j]

                if o == u or o == d or o == l or o == r:
                    return False
        return True

    def absort(self, array):
        r = 0
        for i in range(len(array) - 1):
            if array[i] == array[i+1] \
                    and array[i] != 0:
                array[i] += array[i]
                r += array[i]
                array[i+1] = 0
        i = 0
        j = 0
        # i 指向第一个0，j从i+1向后找，找到第一个非0则和i交换，然后i指向下一个0
        while True:
            if j == len(array) or i == len(array):
                break
            if array[i] != 0:
                i += 1
                #j += 1
            elif array[j] == 0 or j <= i:
                j += 1
            elif array[j] != 0:
                array[i] = array[j]
                array[j] = 0
        return array,r

    def step(self, a):
        done = False
        #s_ = cp.deepcopy(self.state)
        r = 0
        if a == 'u':
            # 每列
            for j in range(self.W):
                tmp = []
                for i in range(self.H):
                    tmp.append(self.state[i][j])
                tmp, sub_r = self.absort(tmp)
                r += sub_r
                for i in range(self.H):
                    self.state[i][j] = tmp[i]

        if a == 'd':
            # 每列
            for j in range(self.W):
                tmp = []
                for i in range(self.H):
                    tmp.append(self.state[self.H - 1 - i][j])
                tmp, sub_r = self.absort(tmp)
                r += sub_r
                for i in range(self.H):
                    self.state[self.H - 1 - i][j] = tmp[i]

        if a == 'l':
            # 每列
            for i in range(self.H):
                tmp = []
                for j in range(self.W):
                    tmp.append(self.state[i][j])
                tmp, sub_r = self.absort(tmp)
                r += sub_r
                for j in range(self.W):
                    self.state[i][j] = tmp[j]

        if a == 'r':
            # 每列
            for i in range(self.H):
                tmp = []
                for j in range(self.W):
                    tmp.append(self.state[i][self.W - 1 - j])
                tmp, sub_r = self.absort(tmp)
                r += sub_r
                for j in range(self.W):
                    self.state[i][self.W - 1 - j] = tmp[j]

        empty = []
        for i in range(self.H):
            for j in range(self.W):
                if self.state[i][j] == 0:
                    empty.append((i,j))

        c = np.random.randint(0,len(empty)-1)
        self.state[empty[c][0]][empty[c][1]] = pow(2, np.random.randint(1,2))

        if self.check_done():
            return self.state,r,True
        else:
            return self.state,r,False


class viz2048():
    def __init__(self, w = 400, h = 400):
        self.env = env2048()
        root = tk.Tk()
        self.viz_w = w
        self.viz_h = h
        self.window = tk.Canvas(
            root,
            width=self.viz_w,
            height=self.viz_w,
            background="white"
        )
        self.draw_background()
        self.draw_num()

        self.window.bind(sequence="<Key>", func=self.processKeyboardEvent)
        self.window.focus_set()
        self.window.pack()
        tk.mainloop()

    def re_draw(self):
        #self.window.delete('line')
        self.window.delete('num')
        self.draw_num()

    def processKeyboardEvent(self, ke):
        print("ke.keysym", ke.keysym)  # 按键别名
        print("ke.char", ke.char)  # 按键对应的字符

        done = False
        if ke.char == 'w':
            s, r, done = self.env.step('u')
        if ke.char == 's':
            s, r, done = self.env.step('d')
        if ke.char == 'a':
            s, r, done = self.env.step('l')
        if ke.char == 'd':
            s, r, done = self.env.step('r')
        if done:
            print('game over !!!!')
        print("ke.keycode", ke.keycode)  # 按键的唯一代码，用于判断按下的是哪个键</class></key></button-1>
        self.re_draw()


    def draw_background(self):
        dw = self.viz_w / self.env.W
        dh = self.viz_h / self.env.H
        # 绘制W列
        for i in range(1, self.env.W):
            self.window.create_line((i*dw,0),(i*dw,self.viz_h),width=2)
        for i in range(1, self.env.H):
            self.window.create_line((0, i*dh), (self.viz_w, i*dh), width=2)

    def draw_num(self):
        dw = self.viz_w / self.env.W
        dh = self.viz_h / self.env.H
        w0 = dw/2
        h0 = dh/2

        for i in range(self.env.W):
            for j in range(self.env.H):
                if self.env.state[j][i] == 0:
                    continue
                self.window.create_text(w0 + i * dw, h0 + j * dh,
                                        text=str(int(self.env.state[j][i])),
                                        tags='num',
                                        font=('宋体',36,'normal'))










        #window.pack()

        #window.create_text(100, 50, text='神农本草经')

        #tk.mainloop()

#    def

    #def step(self, a):
    #   return super.step(a)



def test_rand():
    env = env2048()
    env.reset()

    print(env.state)

    total_r = 0

    while True:
        a = np.random.randint(0,len(env.action_set))
        s,r,done = env.step(env.action_set[a])
        print(env.state)
        total_r += r
        if done:
            print('game over .. r = {}'.format(total_r))
        time.sleep(1)
        print

def test_play_wasd():
    v = viz2048()

if __name__ == '__main__':
    test_play_wasd()
