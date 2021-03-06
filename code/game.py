from base.board import *
from base.player import *

"""
This file is an entry point to the game. It creates a Board and two
Players and iterates the game loop until the game ends.
"""

# create board and players
board = Board()
player1 = ReinforcementLearningStrategy(1)
player2 = MinimaxStrategy(2)

# set player 1 to start
turn = 1

board.draw_board()
# loop game until game end
while not board.check_end():
	if turn == 1:
		player1.play_move(board)
		turn = 2
	elif turn == 2:
		player2.play_move(board)
		turn = 1

	board.draw_board()

# print(board.get_winner())

# board.add_piece(1, 1, 1)
# board.add_piece(2, 0, 0)
# board.add_piece(1, 0, 2)
# board.add_piece(2, 2, 0)
# board.add_piece(1, 1, 0)
# board.add_piece(2, 1, 2)
# board.add_piece(1, 0, 1)
# board.add_piece(2, 2, 1)

# print(player2.select_move(board))