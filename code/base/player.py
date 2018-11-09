import numpy as np

from strategies.minimax import *
from strategies.rl import *

"""
This file contains various player classes that use different strategies 
to select moves.
"""

class Strategy:
	"""
	The base Strategy class is associated with a player (1 or 2). Players
	select a move based on a strategy and then play the move on the board.
	"""

	def __init__(self, player):
		"""Set player number"""
		self.player = player

	def select_move(self, board):
		"""Return selected row, col"""
		return None, None

	def play_move(self, board):
		"""Play selected move on board"""
		row, col = self.select_move(board)
		if row is not None and col is not None:
			board.add_piece(self.player, row, col)

class RandomStrategy(Strategy):
	"""The RandomStrategy selects random moves"""

	def select_move(self, board):
		"""Select random empty space"""
		open_spaces = board.get_open()
		rand_ind = np.random.randint(len(open_spaces))
		return open_spaces[rand_ind]

class MinimaxStrategy(Strategy):
	"""
	The MinimaxStrategy selects the best move from a game tree search using
	the minimax algorithm. This is implemented in strategies/minimax.py.
	"""

	def select_move(self, board):
		"""Select optimal move using minimax on game tree"""
		return minimax_move(self.player, board)

class ReinforcementLearningStrategy(Strategy):
	"""
	The ReinforcementLearningStrategy selects the optimal move by learning an
	optimal policy and selecting a move based on that policy. This is implemented
	in strategies/rl.py.
	"""

	def __init__(self, player):
		"""Learn optimal policy"""
		super().__init__(player)
		self.policy_lookup = dict()
		learn_policy(self.policy_lookup)

	def select_move(self, board):
		"""Select move by looking up board state in policy lookup"""
		return self.policy_lookup
