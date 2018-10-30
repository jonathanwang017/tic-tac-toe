# Train CNN to learn when a player wins

from board import *
from strategy import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


num_samples = 10000
iterations = 60000
batch_size = 10000

# randomly place 3 pieces and check if there is a win
def generate_data():
	train_data = []
	train_labels = []
	for n in range(num_samples):
		board = Board()
		player = RandomStrategy(1)
		# play some pieces (1-5)
		for t in range(np.random.randint(3, 6)):
			player.play_move(board)
		train_data.append(board.grid)
		train_labels.append(board.check_win(1))

	print("%g winning boards"%(np.sum(train_labels)))
	return np.expand_dims(train_data, axis=-1), np.expand_dims(train_labels, axis=-1).astype(int)


def train_model():
	x = tf.placeholder(tf.float32, [None, 3, 3, 1])
	y = tf.placeholder(tf.float32, [None, 1])

	w_conv = tf.Variable(tf.truncated_normal([3, 3, 1, 8], mean=0.5, stddev=0.1))
	b_conv = tf.constant(-2., shape=[8])
	h_conv = tf.nn.relu(tf.nn.conv2d(x, w_conv, strides=[1, 1, 1, 1], padding='VALID') + b_conv)
	
	h_flat = tf.reshape(h_conv, [-1, 8])
	w_fc = tf.constant(20., shape=[8, 1])
	b_fc = tf.constant(-10., shape=[1])
	y_pred = tf.nn.sigmoid(tf.matmul(h_flat, w_fc) + b_fc)

	train_data, train_labels = generate_data()

	loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=y_pred))
	optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
	correct_prediction = tf.equal(tf.math.rint(y_pred), y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())

	batch_xs = np.zeros((batch_size, 3, 3, 1))
	batch_ys = np.zeros((batch_size, 1))

	for i in range(iterations):
		perm = np.arange(num_samples)
		np.random.shuffle(perm)
		for j in range(batch_size):
			batch_xs[j,:,:,:] = train_data[perm[j],:,:,:]
			batch_ys[j,:] = train_labels[perm[j],:]
		
		if i % 100 == 0:
			print("step %d, training loss %g, training accuracy %g"%(i, 
				loss.eval(feed_dict={x: batch_xs, y: batch_ys}),
				accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})))

		optimizer.run(feed_dict={x: batch_xs, y: batch_ys})

	fig = plt.figure()
	fig.suptitle('CNN Filters')
	pos = 1
	for weights in sess.run(w_conv).transpose([3, 0, 1, 2]):
		plt.subplot(3, 3, pos)
		plt.axis('off')
		plt.imshow(weights.squeeze(), cmap='gray')
		pos += 1
	plt.show()

	sess.close()


train_model()


