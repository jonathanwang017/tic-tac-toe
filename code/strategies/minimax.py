import numpy as np
import random

"""
Search game tree and use minimax to determine the best move. This file 
includes counter functionality to compare the search space of minimax 
with and without pruning.
"""

# count total number of boards checked 
# some branches end early due to a player winning
counter = 0

def compute_score(board):
	"""Return inf if player 1 wins and -inf if player 2 wins"""
	if board.check_win(1):
		return float('inf')
	if board.check_win(2):
		return -float('inf')
	return 0

def minimax(player, board):
	"""Select best move using minimax without pruning"""
	global counter
	counter += 1
	# get score at final state
	if board.check_end(): # add depth constraint for non-final scores
		return compute_score(board), None

	# test all available moves in random order
	open_spaces = board.get_open()
	random.shuffle(open_spaces)
	open_scores = []
	for move in open_spaces:
		# try making move
		next_board = board.copy()
		next_board.add_piece(player, move[0], move[1])
		# get score and best opponent move for simulated state
		next_score, next_move = minimax(3 - player, next_board)
		open_scores.append(next_score)
	# return best score and move
	if player == 1:
		return np.max(open_scores), open_spaces[np.argmax(open_scores)]
	elif player == 2:
		return np.min(open_scores), open_spaces[np.argmin(open_scores)]


def alphabeta(player, board, alpha=-float('inf'), beta=float('inf')):
	"""Select best move using minimax with alphabeta pruning"""
	global counter
	counter += 1
	# get score at final state
	if board.check_end(): # TODO: add depth constraint for non-final scores
		return compute_score(board), None

	if player == 1:
		best_score = -float('inf')
		optimizer = lambda score: score > best_score
	elif player == 2:
		best_score = float('inf')
		optimizer = lambda score: score < best_score
	# test all available moves in random order
	open_spaces = board.get_open()
	random.shuffle(open_spaces)
	best_move = None
	for move in open_spaces:
		# try making move
		next_board = board.copy()
		next_board.add_piece(player, move[0], move[1])
		# get score and best opponent move for simulated state
		next_score, next_move = alphabeta(3 - player, next_board, alpha, beta)
		if best_move is None or optimizer(next_score):
			best_score = next_score
			best_move = move
		# prune unnecessary branches
		if player == 1:
			alpha = max(alpha, best_score)
		elif player == 2:
			beta = min(beta, best_score)
		if alpha >= beta:
			break

	return best_score, best_move
	
def minimax_move(player, board, prune=True):
	"""Call appropriate minimax function with or without pruning"""
	if prune:
		best_score, best_move = alphabeta(player, board)
	else:
		best_score, best_move = minimax(player, board) 
	return best_move

def minimax_counter(player, board, prune=True):
	"""Count search space for minimax function with or without pruning"""
	global counter
	counter = 0
	if prune:
		best_score, best_move = alphabeta(player, board)
	else:
		best_score, best_move = minimax(player, board) 
	return counter
