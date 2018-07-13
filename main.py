#%%
import tensorflow as tf
import numpy as np
import cv2
import pylab
from tqdm import tqdm
import chainer
import itertools
train, test = chainer.datasets.get_mnist()
input_data, label_data = train._datasets

def to_plot_img(imgs, n, sz):
    imgs = np.reshape(imgs, (n, sz, sz))
    plot_img = np.zeros([int(n ** 0.5 * sz)] * 2)
    for (i, j), img in zip(itertools.product(range(int(n ** 0.5)), repeat=2), imgs):
        plot_img[i*sz:i*sz+sz, j*sz:j*sz+sz] = img
    return plot_img

#%%
input = tf.placeholder(tf.float32, [None, 784])
h = tf.contrib.slim.fully_connected(input, 300, activation_fn=tf.nn.sigmoid)
h = tf.contrib.slim.dropout(h, 0.5)
output = tf.contrib.slim.fully_connected(h, 784, activation_fn=tf.nn.relu)

with tf.name_scope("optimize"):
    # loss = tf.nn.l2_loss(output - input)
    # loss = tf.losses.cosine_distance(tf.nn.l2_normalize(input, 1), tf.nn.l2_normalize(output, 1), axis=1)
    # loss = tf.keras.backend.binary_crossentropy(input, output)
    loss = tf.losses.mean_squared_error(input, output)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%% test
i = 3
imgs = sess.run(output, feed_dict={input: np.expand_dims(input_data[i], 0)})
pylab.subplot(1, 2, 1)
pylab.imshow(np.reshape(imgs[0], (28, 28)))

pylab.subplot(1, 2, 2)
pylab.imshow(np.reshape(input_data[i], (28, 28)))
pylab.show()

#%% train
batch_n = 4 ** 2
ind = input_data[100:batch_n + 100]
for i in range(300):
    _, imgs, lossv = sess.run([optimizer, output, loss], feed_dict={input: ind})
    pylab.subplot(1, 2, 1)
    pylab.title("epoch: {}, loss: {:.2f}".format(i, lossv))
    pylab.axis("off")
    pylab.imshow(to_plot_img(imgs, batch_n, 28))
    pylab.subplot(1, 2, 2)
    pylab.axis("off")
    pylab.imshow(to_plot_img(ind, batch_n, 28))
    pylab.show()
