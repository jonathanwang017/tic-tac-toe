# Create proxy metric for board score based on winning statistics

from board import *
from strategy import *

# TODO: build and train CNN to learn board score

num_train = 10000


def generate_score_data(num_samples):
	# key: string representing board
	# value: (ties, player1 wins, player2 wins)
	board_score_lookup = dict()

	player1 = RandomStrategy(1)
	player2 = RandomStrategy(2)

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

	for board_string in board_score_lookup.keys():
		board = Board()
		board.expand(board_string)
		data.append(board.grid)

		winners = np.array(board_score_lookup[board_string])
		ties, player1_wins, player2_wins = winners / np.sum(winners)
		labels.append([player1_wins, player2_wins])

	return np.array(data), np.array(labels)

data, labels = generate_score_data(num_train)

				