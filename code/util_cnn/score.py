# Create proxy metric for board score based on winning statistics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from base.board import *
from base.player import *

# TODO: build and train CNN to learn board score

# directory to save plots - relative to analysis.py entry point
images_dir = '../images/'
# directory to save generated data - relative to analysis.py entry point
data_dir = '../data/'

# training parameters
num_train = 100000
num_test = 10000
iterations = 10000

def generate_score_data(num_samples):
	"""Generate dataset of game states with probability of each outcome"""
	try:
		# try loading data if it has already been generated
		data = np.load(data_dir + 'score_data.npy')
		labels = np.load(data_dir + 'score_labels.npy')
	except:
		# key: string representing board
		# value: (ties, player1 wins, player2 wins)
		board_score_lookup = dict()

		player1 = RandomStrategy(1)
		player2 = RandomStrategy(2)

		# num_samples is the number of games simulated
		# not necessarily the data dimension
		for i in range(num_samples):
			board = Board()

			turn = 1
			board_states = []
			# generate random game - record each board state in game
			while not board.check_end():
				board_states.append(board.flatten())
				if turn == 1:
					player1.play_move(board)
					turn = 2
				elif turn == 2:
					player2.play_move(board)
					turn = 1

			# mark game winner for each board state in game
			for board_string in board_states:
				if board_string not in board_score_lookup:
					board_score_lookup[board_string] = [0, 0, 0]
				board_score_lookup[board_string][board.get_winner()] += 1

		data = []
		labels = []

		# format data as nx3x3 (boards) and nx2 (outcome probabilities) arrays
		for board_string in board_score_lookup.keys():
			board = Board()
			board.expand(board_string)
			data.append(board.isolate())

			winners = np.array(board_score_lookup[board_string])
			ties, player1_wins, player2_wins = winners / np.sum(winners)
			labels.append([player1_wins, player2_wins])

		data = np.array(data)
		labels = np.array(labels)

		# save generated data 
		np.save(data_dir + 'score_data.npy', data)
		np.save(data_dir + 'score_labels.npy', labels)

	return data, labels

def generate_validation_data():
	"""Manually create a board to validate model"""
	data = []
	labels = []
	board = Board()
	board.add_piece(1, 1, 1)
	board.add_piece(2, 0, 0)
	board.add_piece(1, 1, 0)
	board.add_piece(2, 2, 1)
	# board.add_piece(1, 1, 2)
	data.append(board.isolate())
	labels.append([1, 0])

	return np.array(data), np.array(labels)

def conv2d(x, w):
	"""Perform convolution of w on x"""
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

def train_score_model():
	"""Train CNN to determine score of a board"""

	# generate data 
	train_data, train_labels = generate_score_data(num_train)
	test_data, test_labels = generate_score_data(num_test)
	val_data, val_labels = generate_validation_data()

	# build model
	x = tf.placeholder(tf.float32, [None, 3, 3, 2])
	y = tf.placeholder(tf.float32, [None, 2])

	w_conv = tf.Variable(tf.truncated_normal([3, 3, 2, 32], stddev=0.1))
	b_conv = tf.Variable(tf.constant(0.1, shape=[32]))
	h_conv = tf.nn.relu(conv2d(x, w_conv) + b_conv)
	
	h_flat = tf.reshape(h_conv, [-1, 32])
	w_fc = tf.Variable(tf.truncated_normal(shape=[32, 2], stddev=0.1))
	b_fc = tf.Variable(tf.constant(0.1, shape=[2]))
	y_pred = tf.add(tf.matmul(h_flat, w_fc), b_fc)
	
	# set up optimization
	loss = tf.reduce_mean(tf.square(y - y_pred))
	optimizer = tf.train.AdadeltaOptimizer(10.).minimize(loss)

	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())

	# save training history
	train_step = []
	train_loss = []
	test_loss = []

	# train model
	for i in range(iterations):
		if i % 100 == 0:
			# evaluate metrics
			train_loss_eval = loss.eval(feed_dict={x: train_data, y: train_labels})
			test_loss_eval = loss.eval(feed_dict={x: test_data, y: test_labels})

			train_step.append(i)
			train_loss.append(train_loss_eval)
			test_loss.append(test_loss_eval)

			print("step %d, training loss %g, test loss %g"%(
				i, train_loss_eval, test_loss_eval))

		# train on all data since batch size is difficult to anticipate
		optimizer.run(feed_dict={x: train_data, y: train_labels})

	# print scores on validation sample
	print(y_pred.eval(feed_dict={x: val_data, y: val_labels}))

	# plot trained weights
	fig = plt.figure()
	fig.suptitle('CNN Filters 1')
	pos = 1
	for weights in sess.run(w_conv).transpose([3, 0, 1, 2]):
		plt.subplot(6, 6, pos)
		plt.axis('off')
		plt.imshow(weights[:, :, 0], cmap='gray')
		pos += 1
	plt.savefig(images_dir + 'score_cnn_weights_1_trained.pdf')
	plt.close()

	fig = plt.figure()
	fig.suptitle('CNN Filters 2')
	pos = 1
	for weights in sess.run(w_conv).transpose([3, 0, 1, 2]):
		plt.subplot(6, 6, pos)
		plt.axis('off')
		plt.imshow(weights[:, :, 1], cmap='gray')
		pos += 1
	plt.savefig(images_dir + 'score_cnn_weights_2_trained.pdf')
	plt.close()

	plt.plot(train_step, train_loss, label='train loss')
	plt.plot(train_step, test_loss, label='test loss')
	plt.title('CNN L2 Loss')
	plt.legend()
	plt.savefig(images_dir + 'score_cnn_learning_history_loss.pdf')
	plt.close()

	sess.close()
				