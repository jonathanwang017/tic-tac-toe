# Loop over game until game ends

from board import *
from strategy import *

board = Board()
player1 = RandomStrategy(1)
player2 = MinimaxStrategy(2)

turn = 1
board.draw_board()
while not board.check_end():
	if turn == 1:
		player1.play_move(board)
		turn = 2
	elif turn == 2:
		player2.play_move(board)
		turn = 1

	board.draw_board()

print(board.get_winner())

# board.add_piece(1, 0, 1)
# board.add_piece(1, 1, 1)
# board.add_piece(1, 2, 0)
# board.add_piece(2, 1, 2)
# board.add_piece(2, 2, 1)

# print(player2.select_move(board))