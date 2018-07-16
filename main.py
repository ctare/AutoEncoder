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
input_test, label_test = test._datasets

def to_plot_img(imgs, n, sz):
    imgs = np.reshape(imgs, (n, sz, sz))
    plot_img = np.zeros([int(n ** 0.5 * sz)] * 2)
    for (i, j), img in zip(itertools.product(range(int(n ** 0.5)), repeat=2), imgs):
        plot_img[i*sz:i*sz+sz, j*sz:j*sz+sz] = img
    return plot_img

#%%
input = tf.placeholder(tf.float32, [None, 784])
reshaped_input = tf.reshape(input, [-1, 28, 28, 1])
conv = tf.contrib.slim.conv2d(reshaped_input, 32, 7, padding="SAME")
pool = tf.contrib.slim.max_pool2d(conv, 2, padding="SAME")
conv = tf.contrib.slim.conv2d(pool, 16, 3, padding="SAME")
pool = tf.contrib.slim.max_pool2d(conv, 2, padding="SAME")
encoder = tf.contrib.slim.conv2d(pool, 1, 3, padding="VALID")

deconv = tf.contrib.slim.conv2d_transpose(encoder, 4, 3, padding="VALID")
upsample = tf.image.resize_nearest_neighbor(deconv, (14, 14))
deconv = tf.contrib.slim.conv2d_transpose(upsample, 16, 3, padding="SAME")
upsample = tf.image.resize_nearest_neighbor(deconv, (28, 28))
deconv = tf.contrib.slim.conv2d_transpose(upsample, 32, 7, padding="SAME")
decoder = tf.contrib.slim.conv2d_transpose(deconv, 1, 3, padding="SAME")
output = tf.contrib.slim.flatten(decoder)

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
for _ in range(50):
    i = int(np.random.uniform(len(input_test)))
    imgs = sess.run(output, feed_dict={input: np.expand_dims(input_test[i], 0)})
    pylab.subplot(1, 2, 1)
    pylab.imshow(np.reshape(imgs[0], (28, 28)))

    pylab.subplot(1, 2, 2)
    pylab.imshow(np.reshape(input_test[i], (28, 28)))
    pylab.show()

#%% train
batch_n = 4 ** 2
for i in range(300):
    target = int(np.random.uniform(len(input_data) - 100))
    ind = input_data[target:batch_n + target]
    _, imgs, encode_images, lossv = sess.run([optimizer, output, encoder, loss], feed_dict={input: ind})
    pylab.subplot(1, 3, 1)
    pylab.title("epoch: {}, loss: {:.2f}".format(i, lossv))
    pylab.axis("off")
    pylab.imshow(to_plot_img(imgs, batch_n, 28))
    pylab.subplot(1, 3, 2)
    pylab.axis("off")
    pylab.imshow(to_plot_img(ind, batch_n, 28))
    pylab.subplot(1, 3, 3)
    pylab.axis("off")
    pylab.imshow(to_plot_img(encode_images, batch_n, 5))
    pylab.show()
