## Tic-Tac-Toe-wards Data Driven Gameplay
In this project, we are examining some different techniques to play tic-tac-toe. In terms of playing the game, we look at basic minimax game tree search, reinforcement learning, and deep reinforcement learning. In addition to just playing the game, we also explore using machine learning to evaluate the game state, specifically identifying if there is a winner and also determining a score for a board.

### Setting Up the Game
Before we can test algorithms to run the game, we need to create the game itself. To do this, we have a `Board` class that represents the game state and has functions that allows players to interact with the board. We also have a `Strategy` class that is a parent class to various strategies that utilize the different algorithms that we will implement to play the game. With these two classes, we can enter a game loop and play tic-tac-toe.

### Identifying a Winning Board
As a warmup, we train a CNN to determine whether a player has won a game. This is a fairly simple deterministic task, but we can nevertheless use data-driven methods to learn this rule. An added benefit is that we can easily verify whether the CNN has properly learned how to perform this task.

##### Generating Winner Data
The first step to a data-driven solution is to get the data. Since this task is symmetric for each player, we will create a dataset from a single player perspective. We randomly place 3 to 5 pieces on the board and check for a win by checking every possible win orientation (8 total). We thus have a dataset of nx3x3x1 features and nx1 labels.

##### Training the Winner CNN
Since this is a task that we can hard code, we can structure the CNN to learn that hard coding. Specifically, we will have 8 filters, each of which will learn one win condition (3 horizontal, 3 vertical, 2 diagonal). We will also fix the convolutional bias as well as the fully connected weights and bias to normalize the outputs for easier prediction. In this model, the only variables will be the 8 filters being learned. The figure below outlines the architecture structure (placeholder). 

![winner_cnn_architecture]

As expected, training this CNN results in filters that match the tic-tac-toe win conditions. By fixing all the parameters besides just the convolutional weights, the filters are learned very cleanly in just 2000 iterations on a batch size of 10000 with Adadelta optimizer and learning rate 10.

| Initial Weights             | Trained Weights              |
|:---------------------------:|:----------------------------:|
|![winner_cnn_weights_initial]|![winner_cnn_weights_trained] |

We will also quickly go over the reasoning behind the architecture in the following diagrams (placeholder).

With the convolution and bias, a filter will output 1 if the input matched that filter and 0 (after ReLU activation) if the input did not. The fully connected layer scales that output and performs an or operation to check if the input matched any filter. That output is then shifted to center around 0, so that taking the sigmoid activation will give 0 if the input did not match any filters and 1 if the input matched any filter.

![winner_activations]
![not_winner_activations]








[winner_cnn_architecture]: diagrams/winner_cnn_architecture.JPG
[winner_cnn_weights_initial]: diagrams/winner_cnn_weights_initial.png
[winner_cnn_weights_trained]: diagrams/winner_cnn_weights_trained.png
[winner_activations]: diagrams/winner_activations.JPG
[not_winner_activations]: diagrams/not_winner_activations.JPG
