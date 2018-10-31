# Search game tree and use minimax to determine best move
# Includes counter functionality to compare search space of minimax with and without pruning
from board import *

# count total number of boards checked 
# some branches end early due to a player winning
counter = 0

def compute_score(board):
	if board.check_win(1):
		return float('inf')
	if board.check_win(2):
		return -float('inf')
	return 0


def minimax(player, board):
	global counter
	counter += 1
	# get score at final state
	if board.check_end(): # add depth constraint for non-final scores
		return compute_score(board), None

	# test all available moves
	open_spaces = board.get_open()
	open_scores = []
	for move in open_spaces:
		# try making move
		next_board = Board()
		next_board.grid = board.grid.copy()
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
	# test all available moves
	open_spaces = board.get_open()
	best_move = None
	for move in open_spaces:
		# try making move
		next_board = Board()
		next_board.grid = board.grid.copy()
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
	if prune:
		best_score, best_move = alphabeta(player, board)
	else:
		best_score, best_move = minimax(player, board) 
	return best_move

def minimax_counter(player, board, prune=True):
	global counter
	counter = 0
	if prune:
		best_score, best_move = alphabeta(player, board)
	else:
		best_score, best_move = minimax(player, board) 
	return counter
