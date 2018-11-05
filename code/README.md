### Code Structure
This folder contains all of the code for the tic-tac-toe game and the various learning tasks. Due to the packaging structure of the code, the two python files in the root code/ directory (`game.py` and `analysis.py`) are the entry points to the rest of the code. Here is the breakdown of the code structure:

- `game.py` - main entrypoint to game. creates board and players and runs game loop.
- `analysis.py` - contains functions that simulates games or parts of game to examine results.
- `base/` - contains the foundational classes for the game
	- `board.py` - primary game object that contains game state information and functions to interact with the game.
	- `player.py` - various player strategies that use different algorithms to play the game.
- `strategies/` - contains various strategies for selecting a move.
	- `minimax.py` - implements the minimax algorithm with and without alphabeta pruning to search the game tree for the best move.
	- `rl.py` - implements policy and value iteration to learn the optimal policy to select the best move.
- `util_cnn/` - contains the training functions for the tasks of identifying a winner and scoring a board.
	- `winner.py` - generates random game data and trains a CNN to identify if a player has won.
	- `score.py` - generates random winner data and trains a CNN to identify the probability of a player winning from a board state, thus scoring the board.

