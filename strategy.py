# Represent each player as a strategy to interact with board

from board import *
from minimax import *

class Strategy:
	def __init__(self, player):
		self.player = player

	def select_move(self, board):
		return None, None

	def play_move(self, board):
		row, col = self.select_move(board)
		if row is not None and col is not None:
			board.add_piece(self.player, row, col)

class RandomStrategy(Strategy):
	def select_move(self, board):
		open_spaces = board.get_open()
		rand_ind = np.random.randint(len(open_spaces))
		return open_spaces[rand_ind]

class MinimaxStrategy(Strategy):
	def select_move(self, board):
		return minimax_move(self.player, board)
