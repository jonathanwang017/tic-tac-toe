## Tic-Tac-Toe-wards Data Driven Gameplay

### Introduction
Tic-tac-toe is a fairly simple game, but it can be used to demonstrate various methods for creating computer player strategies. This project will look at several aspects of the game - determining a winner, evaluating a board, and selecting a move.

### Winning Boards
For many games, it can be difficult to determine whether a game state has a winner. The most naive method would be to perform some brute force checks. This could be improved by some noticeable patterns. However, an interesting data driven approach would be to train a CNN to identify endgame states. 

### Scoring Boards
For gameplay strategies such as minimax, we need to know what the score of a board is. If we search the game tree to full depth, then we will know the winner, and thus have a score. However, if we want to restrict the depth of the game tree, we will need some way to score intermediate board states. We can simulate games to determine the chance of a player winning a game from a board state, and train a CNN to predict this probability based on the board.

### Selecting Moves
In tic-tac-toe, if both players play optimally, then the game should alway result in a tie. However, if one player is playing optimally against a random opponent, the optimal player should win often and never lose. We will skip hard-coded strategies for a computer player, since it is infeasible to cover every board state, even with symmetry generalizations. Instead, we will start will state space search using game trees, followed by reinforcement learning. Lastly, we will look at training an RNN or using DeepRL to identify the best moves.