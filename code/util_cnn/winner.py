import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from base.board import *
from base.player import *

"""
This file contains functions to generate random game data for one player
and train a CNN to determine if a player has won.
"""

# directory to save plots - relative to analysis.py entry point
images_dir = '../images/'
# directory to save models - relative to analysis.py entry point
model_dir = '../models/'

# training parameters
num_train = 10000
num_test = 1000
num_val = 10
iterations = 2000
batch_size = 10000

def generate_winner_data(num_samples, min_pieces):
	"""Generate labelled random game data""" 
	data = []
	labels = []
	for n in range(num_samples):
		board = Board()
		player = RandomStrategy(1)
		# play some pieces randomly
		for t in range(np.random.randint(min_pieces, 6)):
			player.play_move(board)
		data.append(board.grid)
		# check if player has won
		labels.append(board.check_win(1))

	# check data distribution
	print("%g winning boards"%(np.sum(labels)))
	return np.expand_dims(data, axis=-1), np.expand_dims(labels, axis=-1).astype(int)

def conv2d(x, w):
	"""Perform convolution of w on x"""
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

def train_winner_model():
	"""Train CNN to determine if a player has won"""

	# generate data 
	# NOTE: test data has different distribution since train data boards
	# have minimum of 3 pieces. weights learned from train data generalize
	# to test data even with different distribution
	train_data, train_labels = generate_winner_data(num_train, 3)
	test_data, test_labels = generate_winner_data(num_test, 0)

	# build model
	# convolutional filters should learn each winning pattern. convolution
	# identifies winning patterns and fully connected layer normalizes and 
	# checks whether there are any winning patterns
	x = tf.placeholder(tf.float32, [None, 3, 3, 1])
	y = tf.placeholder(tf.float32, [None, 1])

	w_conv = tf.Variable(tf.truncated_normal([3, 3, 1, 8], mean=0.5, stddev=0.1))
	b_conv = tf.constant(-2., shape=[8])
	h_conv = tf.nn.relu(conv2d(x, w_conv) + b_conv)
	
	h_flat = tf.reshape(h_conv, [-1, 8])
	w_fc = tf.constant(20., shape=[8, 1])
	b_fc = tf.constant(-10., shape=[1])
	y_pred = tf.nn.sigmoid(tf.matmul(h_flat, w_fc) + b_fc)

	# set up optimization
	loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y, y_pred))
	optimizer = tf.train.AdadeltaOptimizer(10.).minimize(loss)
	correct_prediction = tf.equal(tf.math.rint(y_pred), y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())

	batch_xs = np.zeros((batch_size, 3, 3, 1))
	batch_ys = np.zeros((batch_size, 1))

	# plot initial weights
	fig = plt.figure()
	fig.suptitle('CNN Filters')
	pos = 1
	for weights in sess.run(w_conv).transpose([3, 0, 1, 2]):
		plt.subplot(3, 3, pos)
		plt.axis('off')
		plt.imshow(weights.squeeze(), cmap='gray')
		pos += 1
	plt.savefig(images_dir + 'winner_cnn_weights_initial.pdf')
	plt.close()

	# save training history
	train_step = []
	train_accuracy = []
	train_loss = []
	test_accuracy = []
	test_loss = []

	# train model
	for i in range(iterations):
		perm = np.arange(num_train)
		np.random.shuffle(perm)
		for j in range(batch_size):
			batch_xs[j,:,:,:] = train_data[perm[j],:,:,:]
			batch_ys[j,:] = train_labels[perm[j],:]
		
		if i % 100 == 0:
			# evaluate metrics
			train_acc_eval = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
			train_loss_eval = loss.eval(feed_dict={x: batch_xs, y: batch_ys})
			test_acc_eval = accuracy.eval(feed_dict={x: test_data, y: test_labels})
			test_loss_eval = loss.eval(feed_dict={x: test_data, y: test_labels})

			train_step.append(i)
			train_accuracy.append(train_acc_eval)
			train_loss.append(train_loss_eval)
			test_accuracy.append(test_acc_eval)
			test_loss.append(test_loss_eval)

			print("step %d, training loss %g, training accuracy %g, test accuracy %g"%(
				i, train_loss_eval, train_acc_eval, test_acc_eval))

		optimizer.run(feed_dict={x: batch_xs, y: batch_ys})

	# save model
	tf.saved_model.simple_save(sess, model_dir + 'winner_model', 
		inputs={'x': x, 'y': y}, outputs={'y_pred': y_pred})

	# plot trained weights
	fig = plt.figure()
	fig.suptitle('CNN Filters')
	pos = 1
	for weights in sess.run(w_conv).transpose([3, 0, 1, 2]):
		plt.subplot(3, 3, pos)
		plt.axis('off')
		plt.imshow(weights.squeeze(), cmap='gray')
		pos += 1
	plt.savefig(images_dir + 'winner_cnn_weights_trained.pdf')
	plt.close()

	# plot learning history
	plt.plot(train_step, train_accuracy, label='train accuracy')
	plt.plot(train_step, test_accuracy, label='test accuracy')
	plt.title('CNN Accuracy')
	plt.legend()
	plt.savefig(images_dir + 'winner_cnn_learning_history_accuracy.pdf')
	plt.close()

	plt.plot(train_step, train_loss, label='train loss')
	plt.plot(train_step, test_loss, label='test loss')
	plt.title('CNN Cross Entropy Loss')
	plt.legend()
	plt.savefig(images_dir + 'winner_cnn_learning_history_loss.pdf')
	plt.close()

	sess.close()

def load_winner_model():
	"""Load graph of trained winner model"""
	with tf.Session(graph=tf.Graph()) as sess:
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 
			model_dir + 'winner_model')
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name('x:0')
		y = graph.get_tensor_by_name('y:0')
		y_pred = graph.get_tensor_by_name('y_pred:0')

def predict_winner_model(data=None, labels=None):
	"""Use trained winner model to determine if player has won"""
	
	# generate sample data if none supplied
	if data is None and labels is None:
		data, labels = generate_winner_data(num_val, 0)

	# use tf predictor api on trained model
	predict_fn = tf.contrib.predictor.from_saved_model(model_dir + 'winner_model')
	y_pred = predict_fn({'x': data, 'y': labels})['y_pred']
	for board, label, pred in zip(data, labels, y_pred):
		print(board.squeeze(), label, pred)
