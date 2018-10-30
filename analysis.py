# Explore and analyze different game scenarios

from board import *
from strategy import *

import matplotlib.pyplot as plt
import numpy as np

def simulate_winner(player1, player2):
	winners = []
	for i in range(1000):
		if i % 100 == 0:
			print(i)
		board = Board()

		turn = 1
		while not board.check_end():
			if turn == 1:
				player1.play_move(board)
				turn = 2
			elif turn == 2:
				player2.play_move(board)
				turn = 1

		winners.append(board.get_winner())
	plt.hist(winners)
	plt.show()

def random_random():
	simulate_winner(RandomStrategy(1), RandomStrategy(2))

def minimax_random():
	simulate_winner(MinimaxStrategy(1), RandomStrategy(2))

def random_minimax():
	simulate_winner(RandomStrategy(1), MinimaxStrategy(2))

random_random()