## Tic-Tac-Toe-wards Data Driven Gameplay
In this project, we are examining some different techniques to play tic-tac-toe. In terms of playing the game, we look at basic minimax game tree search, reinforcement learning, and deep reinforcement learning. In addition to just playing the game, we also explore using machine learning to evaluate the game state, specifically identifying if there is a winner and also determining a score for a board.

### Setting Up the Game
Before we can test algorithms to run the game, we need to create the game itself. To do this, we have a `Board` class that represents the game state and has functions that allows players to interact with the board. We also have a `Strategy` class that is a parent class to various strategies that utilize the different algorithms that we will implement to play the game. With these two classes, we can enter a game loop and play tic-tac-toe.

### Identifying a Winning Board
As a warmup, we train a CNN to determine whether a player has won a game. This is a fairly simple deterministic task, but we can nevertheless use data-driven methods to learn this rule. An added benefit is that we can easily verify whether the CNN has properly learned how to perform this task.

#### Generating Winner Data
The first step to a data-driven solution is to get the data. Since this task is symmetric for each player, we will create a dataset from a single player perspective. We randomly place 3 to 5 pieces on the board and check for a win by checking every possible win orientation (8 total). We thus have a dataset of nx3x3x1 features and nx1 labels.

#### Training the Winner CNN
Since this is a task that we can hard code, we can structure the CNN to learn that hard coding. Specifically, we will have 8 filters, each of which will learn one win condition (3 horizontal, 3 vertical, 2 diagonal). We will also fix the convolutional bias as well as the fully connected weights and bias to normalize the outputs for easier prediction. In this model, the only variables will be the 8 filters being learned. The figure below outlines the architecture structure (placeholder). 

![winner_cnn_architecture]

As expected, training this CNN results in filters that match the tic-tac-toe win conditions. By fixing all the parameters besides just the convolutional weights, the filters are learned very cleanly in just 2000 iterations on a batch size of 10000 with Adadelta optimizer and learning rate 10 and sigmoid cross entropy loss.

| Initial Weights             | Trained Weights              |
|:---------------------------:|:----------------------------:|
|![winner_cnn_weights_initial]|![winner_cnn_weights_trained] |

We will also quickly go over the reasoning behind the architecture in the following diagrams (placeholder).

With the convolution and bias, a filter will output 1 if the input matched that filter and 0 (after ReLU activation) if the input did not. The fully connected layer scales that output and performs an or operation to check if the input matched any filter. That output is then shifted to center around 0, so that taking the sigmoid activation will give 0 if the input did not match any filters and 1 if the input matched any filter.

![winner_activations]
![not_winner_activations]

### Scoring a Board
A more challenging task is quantifying a game state. At the end of a game, the score can be decided simply by which player won or if there was a tie. However, an intermediate board state is much harder to score. A proxy value we decided on is the statistics of a player winning from a board state. We can then train a CNN to predict this score for each player.

#### Generating Score Data
This task is no longer symmetric for the players, so we will simulate games between two players. For each game, we collect all the board states of the game and label them with the final winner. This is stored in a dictionary of board:(ties, player 1 wins, player 2 wins), thus tracking the witnessed outcomes of each board state. For the prediction task, we split the board into two channels, one for each player. We normalize the outcome statistics by dividing by the total outcomes for a board and only consider the outcomes where a player wins. The resulting dataset consists of nx3x3x2 features and nx2 labels.

#### Training the Score CNN
As this prediction task is less structured that the winner task, we will not fix any of the weight parameters. We will, however, use a similar architecture to the winner CNN, with a convolutional layer with ReLU into a fully connected layer. Instead of 8 3x3x1 filters, we will have 32 3x3x2 filters. Each filter has one channel for each player channel. Since we are predicting a continuous variable rather than a classification, we will also not have any activation after the fully connected layer. 

The resulting trained weights show an interesting, but predictable pattern. The channels of the filters are mostly inverses of each other, since at the end of the game, tic-tac-toe is zero-sum. Also, many of the filters match the win conditions as found in the winner CNN filters, which makes sense since game winning boards gives a high score.

| Channel 1 Weights            | Channel 2 Weights             |
|:----------------------------:|:-----------------------------:|
|![score_cnn_weights_1_trained]|![score_cnn_weights_2_trained] |

For the model training, we use the Adadelta optimzer with learning rate 10 again. Our loss function is L2 loss, and we optimize for 10000 iterations over the full training dataset.

We can simulate a game, and examine the predicted scores to determine whether the model learned a reasonable scoring function. In the images, a black space is empty; gray is a player 1 piece; and white is a player 2 piece. At the very start, the empty board state has a higher score for player 1 since the first player has a better chance of winning. On turn 4, when player 1 is on the verge of winning, player 1 has a very high score. However, the player 1 score drops, when the random move doesn't win, and the player 2 score jumps up when it blocks the player 1 win. The game ends with a tie, where both players have low score, since neither can win.

| Turn | Board State | Player 1 Score | Player 2 Score |
|:----:|:-----------:|:--------------:|:--------------:|
| 0    | ![turn0]    | 0.568          | 0.291          |
| 1    | ![turn1]    | 0.576          | 0.281          |
| 2    | ![turn2]    | 0.539          | 0.313          |
| 3    | ![turn3]    | 0.573          | 0.337          |
| 4    | ![turn4]    | 0.744          | 0.117          |
| 5    | ![turn5]    | 0.485          | 0.314          |
| 6    | ![turn6]    | 0.214          | 0.510          |
| 7    | ![turn7]    | 0.063          | 0.201          |
| 8    | ![turn8]    | 0.116          | 0.403          |
| 9    | ![turn9]    | 0.127          | 0.028          |


[winner_cnn_architecture]: diagrams/winner_cnn_architecture.JPG
[winner_cnn_weights_initial]: diagrams/winner_cnn_weights_initial.png
[winner_cnn_weights_trained]: diagrams/winner_cnn_weights_trained.png
[winner_activations]: diagrams/winner_activations.JPG
[not_winner_activations]: diagrams/not_winner_activations.JPG
[score_cnn_weights_1_trained]: diagrams/score_cnn_weights_1_trained.png
[score_cnn_weights_2_trained]: diagrams/score_cnn_weights_2_trained.png

[turn0]: images/score_example/turn0.png
[turn1]: images/score_example/turn1.png
[turn2]: images/score_example/turn2.png
[turn3]: images/score_example/turn3.png
[turn4]: images/score_example/turn4.png
[turn5]: images/score_example/turn5.png
[turn6]: images/score_example/turn6.png
[turn7]: images/score_example/turn7.png
[turn8]: images/score_example/turn8.png
[turn9]: images/score_example/turn9.png
