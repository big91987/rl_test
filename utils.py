import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import numpy as np
import cv2
from queue import Queue
import copy
import time

# 输入 N个3通道的图片array
# 输出：一个array 形状 （84 84 N）
# 步骤： 1 resize ==>（84 84 3）[0-255]
#       2 gray   ==> (84 84 1) [0-255]
#       3 norm   ==> (84 84 1) [0.0-1.0]
#       4 concat ===>(84 84 N) [0.0-1.0]
def imgbuffer_pergress(imgbuffer):
    img_list = []
    for img in imgbuffer:
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmp = cv2.resize(src=tmp, dsize=(84, 84))
        cv2.normalize(tmp, tmp, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=)
        tmp = np.expand_dims(tmp, len(tmp.shape))
        #print ('tmp_shape = '+ str(tmp.shape))
        img_list.append(tmp)
    ret =  np.concatenate(tuple(img_list), axis=2)
    #print('ret_shape = ' + str(ret.shape))
    return ret

    #pass

def test():
    env = gym.make('Breakout-v4')
    env.seed(1)  # reproducible
    #env = env.unwrapped
    N_F = env.observation_space.shape[0]  # 状态空间的维度
    N_A = env.action_space.n  # 动作空间的维度

    #img_buffer = Queue(maxsize=4)
    img_buffer = []
    img_buffer_size = 4

    s = env.reset()

    max_loop = 100000

    for i in range(max_loop):
        a = np.random.randint(0, N_A - 1)
        s_, r, done, info = env.step(a)
        #env.render()

        if len(img_buffer) < img_buffer_size:
            img_buffer.append(s_)
            continue
        else:
            img_buffer.pop(0)
            img_buffer.append(s_)

        img_input = imgbuffer_pergress(img_buffer)
        print ('img_input_shape = ' + str(img_input.shape))
        plt.imshow(np.uint8(img_input[:,:,0] * 255))
        plt.show()

    #with tf.Session() as sess:
    #    for i in range(max_loop):
    #        a = np.random.randint(0, N_A - 1)
    #        s_, r, done, info = env.step(a)

    #        if len(img_buffer) < img_buffer_size:
    #            img_buffer.append(s_)
    #            continue
    #        else:
    #            img_buffer.pop(0)
    #            img_buffer.append(s_)

            #tmp_buffer = copy.deepcopy()
            #gray_img_list = []
            #for img in img_buffer:
                # 以下代码用tensorflow处理，不太爽
                #img_data = tf.image.convert_image_dtype(img, dtype=tf.float32)
                #img_data = tf.image.resize_images(img_data, [84, 84], method=0)
                #img_data = tf.image.rgb_to_grayscale(img_data)
                #img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
                #gray_img_list.append(img_data)
                #resize_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
                #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                #nor_img = cv2.normalize(img,img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                #cv2.imshow('123', img)
                #cv2.waitKey(0)
                #time.sleep(1)
                #cv2.destroyAllWindows()
                #plt.imshow(np.uint8(nor_img * 255))
                #plt.show()

            #gray_img = tf.concat(gray_img_list, 2)

            #print(gray_img.shape)

            #overlap_img = tf.image.convert_image_dtype(gray_img, dtype=tf.float32)
            #overlap_img = tf.reduce_sum(overlap_img, axis=2)
            #overlap_img = np.asarray(sess.run(overlap_img)/1.1, dtype='uint8')

            #test_img = np.asarray(sess.run(gray_img[:,:,1]), dtype='uint8')
            #plt.imshow(test_img, cmap='gray')
            #plt.show()


            #img_data = tf.image.convert_image_dtype(s_, dtype=tf.float32)

            # 将表示一张图像的三维矩阵按照jpeg的格式重新编码并保存。可得到与原图一样的图像。
            #img_data = tf.image.resize_images(img_data, [84, 84], method=0)
            #img_data = tf.image.rgb_to_grayscale(img_data)
            #img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

            #encode_image = tf.image.encode_jpeg(img_data)

            #img = np.asarray(sess.run(img_data), dtype='uint8')

            #print (' i = ' + str(i))
            #plt.imshow(img[:, :, 0], cmap='gray')
            #plt.show()

            #with tf.gfile.GFile("./123.jpeg", "wb") as f:
            #    f.write(encode_image.eval())
    #pass

if __name__ == '__main__':
    test()
