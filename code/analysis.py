import matplotlib.pyplot as plt
import numpy as np

from base.board import *
from base.player import *
from strategies.minimax import *
from strategies.rl import *
from util_cnn.winner import *
from util_cnn.score import *

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
	plt.hist(winners, range=(0, 2))
	plt.title(player1.strategy_name + ' vs ' + player2.strategy_name)
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

def rl_random():
	"""Compare player 1 reinforcement learning and player 2 random"""
	simulate_winner(ReinforcementLearningStrategy(1), RandomStrategy(2))

def random_rl():
	"""Compare player 1 random and player 2 reinforcement learning"""
	simulate_winner(RandomStrategy(1), ReinforcementLearningStrategy(2))

def minimax_alphabeta():
	"""Compare and plot the search space of minimax with and without pruning""" 
	board = Board()
	minimax_count = minimax_counter(1, board, prune=False)
	alphabeta_count = minimax_counter(1, board)
	plt.bar([0, 1], [minimax_count, alphabeta_count])
	plt.xticks([0, 1], ('Minimax', 'Alphabeta'))
	plt.show()

def test_winner_cnn():
	"""Identify if player has won board on sample data"""
	predict_winner_model()

def test_score_cnn():
	"""Identify chances of each player winning board on sample data"""
	predict_score_model()

def test_rl_policy():
	"""Examine policy values of board states in a game"""
	board = Board()
	player1 = ReinforcementLearningStrategy(1)
	player2 = ReinforcementLearningStrategy(2)
	turn = 1
	while not board.check_end():
		if turn == 1:
			board.draw_board()
			print(visualize_policy(player1.policy_lookup, board, turn))
			player1.play_move(board)
			turn = 2
		elif turn == 2:
			board.draw_board()
			print(visualize_policy(player2.policy_lookup, board, turn))
			player2.play_move(board)
			turn = 1
