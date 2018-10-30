## Tic-Tac-Toe-wards Data Driven Gameplay

### Introduction
Tic-tac-toe is a fairly simple game, but it can be used to demonstrate various methods for creating computer player strategies. This project will look at two aspects of the game - determining a winner and selecting a move.

### Winning Boards
For many games, it can be difficult to determine whether a game state has a winner. The most naive method would be to perform some brute force checks. This could be improved by some noticeable patterns. However, an interesting data driven approach would be to train a CNN to identify endgame states. 

### Selecting Moves
In tic-tac-toe, if both players play optimally, then the game should alway result in a tie. However, if one player is playing optimally against a random opponent, the optimal player should win often and never lose. We will skip hard-coded strategies for a computer player, since it is infeasible to cover every board state, even with symmetry generalizations. Instead, we will start will state space search using game trees, followed by reinforcement learning. Lastly, we will look at training an RNN to identify the best moves.