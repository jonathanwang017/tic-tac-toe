import numpy as np
import matplotlib.pyplot as plt

class Board:
	"""
	The Board class represents the game object. It includes functions
	that allow a player to interact with the game and to get information
	about the game state.
	"""

	def __init__(self):
		"""Initialize 3x3 board to zeros (empty)"""
		self.grid = np.zeros((3, 3)).astype(int)

	def draw_board(self):
		"""Display board values"""
		print(self.grid)

	def plot_board(self, filename=None):
		"""Plot board and save if filename is supplied"""
		plt.figure(figsize=(1,1))
		plt.axis('off')
		# convert grid to [-1, 1] where -1 is empty, 
		# 0 is player 1 and 1 is player 2
		plt.imshow(self.grid - 1, cmap='gray', vmin=-1, vmax=1)
		if filename is None:
			plt.show()
		else:
			plt.savefig(filename)
			plt.close()

	def flatten(self):
		"""Return board as string representation"""
		return np.array2string(self.grid.flatten(), separator='')[1:-1]

	def expand(self, string):
		"""Convert string representation to board grid"""
		for i in range(3):
			for j in range(3):
				self.grid[i, j] = string[i * 3 + j]

	def isolate(self):
		"""Convert board to two channels - one for each player"""
		return (np.arange(2) == self.grid[..., None] - 1).astype(int)

	def copy(self):
		"""Create a copy of this board"""
		new_board = Board()
		new_board.grid = self.grid.copy()
		return new_board

	def clear_board(self):
		"""Reset 3x3 board to zeros (empty)"""
		self.grid = np.zeros((3, 3)).astype(int)

	def get_open(self):
		"""Find empty spaces on board"""
		open_spaces = np.where(self.grid == 0)
		return list(zip(open_spaces[0], open_spaces[1]))

	def add_piece(self, player, row, col):
		"""Add player piece to board at (row, col)"""
		# check valid input
		assert player == 1 or player == 2, 'Invalid player'
		assert row >= 0 and row <= 2, 'Invalid row'
		assert col >= 0 and row <= 2, 'Invalid column'
		# check if piece already exists at location
		assert self.grid[row, col] == 0, 'Piece already exists at location'
		self.grid[row, col] = player

	def check_win(self, player):
		"""Check if player has won"""
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
		"""Check if game is over"""
		# check for win
		if self.check_win(1) or self.check_win(2):
			return True
		# check for tie
		empty_spaces = (self.grid == 0).astype(int)
		if np.sum(empty_spaces) == 0:
			return True
		return False

	def get_winner(self):
		"""Return winning player or tie"""
		assert self.check_end(), 'Game is not over yet'
		if self.check_win(1):
			return 1
		if self.check_win(2):
			return 2
		return 0
