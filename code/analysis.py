import matplotlib.pyplot as plt
import numpy as np

from base.board import *
from base.player import *
from strategies.minimax import *
from util_cnn import winner
from util_cnn import score

"""
This file contains various functions to analyze different game scenarios.
For instance, we can compare the results of different strategies or examine
the advantage of going first, etc.
"""

def simulate_winner(player1, player2):
	"""Simulate and plot results of games between two players"""
	winners = []
	# play 1000 games
	for i in range(1000):
		if i % 100 == 0:
			print(i)
		board = Board()

		turn = 1
		# enter game loop
		while not board.check_end():
			if turn == 1:
				player1.play_move(board)
				turn = 2
			elif turn == 2:
				player2.play_move(board)
				turn = 1

		winners.append(board.get_winner())
	# plot results of games
	plt.hist(winners)
	plt.show()

def random_random():
	"""Compare two random strategies"""
	simulate_winner(RandomStrategy(1), RandomStrategy(2))

def minimax_random():
	"""Compare player 1 minimax and player 2 random"""
	simulate_winner(MinimaxStrategy(1), RandomStrategy(2))

def random_minimax():
	"""Compare player 1 random and player 2 minimax"""
	simulate_winner(RandomStrategy(1), MinimaxStrategy(2))

def minimax_alphabeta():
	"""Compare and plot the search space of minimax with and without pruning""" 
	board = Board()
	minimax_count = minimax_counter(1, board, prune=False)
	alphabeta_count = minimax_counter(1, board)
	plt.bar([0, 1], [minimax_count, alphabeta_count])
	plt.xticks([0, 1], ('Minimax', 'Alphabeta'))
	plt.show()

def train_winner_cnn():
	winner.train_model()
