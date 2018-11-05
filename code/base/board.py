# Board class represents game object

import numpy as np

class Board:
	def __init__(self):
		self.grid = np.zeros((3, 3)).astype(int)

	def draw_board(self):
		print(self.grid)

	def flatten(self):
		return np.array2string(self.grid.flatten(), separator='')[1:-1]

	def expand(self, string):
		for i in range(3):
			for j in range(3):
				self.grid[i, j] = string[i * 3 + j]

	def clear_board(self):
		self.grid = np.zeros((3, 3)).astype(int)

	def get_open(self):
		open_spaces = np.where(self.grid == 0)
		return list(zip(open_spaces[0], open_spaces[1]))

	def add_piece(self, player, row, col):
		# check valid input
		assert player == 1 or player == 2, 'Invalid player'
		assert row >= 0 and row <= 2, 'Invalid row'
		assert col >= 0 and row <= 2, 'Invalid column'
		# check if piece already exists at location
		assert self.grid[row, col] == 0, 'Piece already exists at location'
		self.grid[row, col] = player

	def check_win(self, player):
		# check valid player
		assert player == 1 or player == 2, 'Invalid player'
		# isolate player pieces 
		player_pieces = (self.grid == player).astype(int)
		# check verticals
		if 3 in np.sum(player_pieces, axis=0):
			return True
		# check horizontals
		if 3 in np.sum(player_pieces, axis=1):
			return True
		# check forward diagonal
		if np.sum(np.diag(player_pieces)) == 3:
			return True
		# check reverse diagonal
		if np.sum(np.diag(np.fliplr(player_pieces))) == 3:
			return True
		return False

	def check_end(self):
		# check for win
		if self.check_win(1) or self.check_win(2):
			return True
		# check for tie
		empty_spaces = (self.grid == 0).astype(int)
		if np.sum(empty_spaces) == 0:
			return True
		return False

	def get_winner(self):
		assert self.check_end(), 'Game is not over yet'
		if self.check_win(1):
			return 1
		if self.check_win(2):
			return 2
		return 0

