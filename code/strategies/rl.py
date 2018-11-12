import numpy as np
import pickle
import random

from base.board import *

"""
This file contains functions to learn an optimal policy using Q-learning.
The game is played randomly and policy values are updated according to the
rewards, where a reward of +1 is given for a player 1 win and a reward of -1
for a player 2 win. All other states have a reward of 0. The result is an action
policy as well as a sort of data-driven score for each state and action.

The game between two reinforcement strategy players will almost always
be the same since the policy is saved and loaded and there is almost no
randomness in the policy exploitation function. We can either re-learn the
policy every time or add some randomness (epsilon) when selecting policy
exploitation moves.
"""

# directory to save models - relative to analysis.py entry point
model_dir = '../models/rl_model/'

discount = 1
learning_rate = 0.01
epsilon = 0.8
iterations = 50000

def learn_policy(policy_lookup):
	"""Play games to learn optimal policy"""
	try:
		# try loading policy lookup if already learned
		with open(model_dir + 'rl_policy.pickle', 'rb') as model:
			loaded_policy = pickle.load(model)
			for state in loaded_policy.keys():
				policy_lookup[state] = loaded_policy[state]
	except:
		for i in range(iterations):
			board = Board()
			turn = 1
			print(visualize_policy(policy_lookup, board, turn))
			# play game and update policy values based on reward
			while not board.check_end():
				state = board.flatten()
				# select policy based move or random move with epsilon probability
				if np.random.random() > epsilon:
					action = exploit_policy(policy_lookup, board, turn)
				else:
					action = explore_policy(board)
				board.add_piece(turn, action[0], action[1])
				next_state = board.flatten()
				reward = get_reward(board)
				update_policy_values(
					policy_lookup, board, state, turn, action, next_state, reward)
				turn = 3 - turn
		# pickle policy lookup
		with open(model_dir + 'rl_policy.pickle', 'wb') as model:
			pickle.dump(policy_lookup, model)
	
def explore_policy(board):
	"""Select random move"""
	open_spaces = board.get_open()
	rand_ind = np.random.randint(len(open_spaces))
	return open_spaces[rand_ind]
	
def exploit_policy(policy_lookup, board, player):
	"""Select best move based on policy"""
	state = board.flatten()
	best_value = None
	best_action = None
	# look up policy action values for all possible moves (shuffled)
	actions_list = board.get_open()
	random.shuffle(actions_list)
	for action in actions_list:
		value = get_policy_value(policy_lookup, state, player, action)
		if (best_action is None or 
			player == 1 and value > best_value or
			player == 2 and value < best_value):
			best_value = value
			best_action = action
	return best_action

def get_reward(board):
	"""Return reward for board state"""
	if board.check_end():
		# +1 if player 1 wins
		if board.get_winner() == 1:
			return 1
		# -1 if player 2 wins
		elif board.get_winner() == 2:
			return -1
		else:
			return 0
	else:
		return 0

def update_policy_values(policy_lookup, board, state, player, action, next_state, reward):
	"""Update policy values with Bellman equation"""
	
	# compute total expected reward including future rewards
	if board.check_end():
		expected = reward
	else:
		if player == 1:
			expected = reward + discount * min_value(policy_lookup, next_state, 2)
		elif player == 2:
			expected = reward + discount * max_value(policy_lookup, next_state, 1)
	# get current policy action value
	policy_value = get_policy_value(policy_lookup, state, player, action)
	# update policy action value
	policy_lookup[(state, player)][action] += learning_rate * (expected - policy_value)

def get_policy_actions(policy_lookup, state, player):
	"""Get policy state actions if they exist"""
	if (state, player) not in policy_lookup:
		policy_lookup[(state, player)] = dict()
	return policy_lookup[(state, player)]

def get_policy_value(policy_lookup, state, player, action):
	"""Get policy action value if it exists"""
	if action not in get_policy_actions(policy_lookup, state, player):
		policy_lookup[(state, player)][action] = 0
	return policy_lookup[(state, player)][action]

def min_value(policy_lookup, state, player):
	"""Return minimum policy action value for a state"""
	action_values = list(get_policy_actions(policy_lookup, state, player).values())
	if action_values:
		return np.min(action_values)
	return 0

def max_value(policy_lookup, state, player):
	"""Return maximum policy action value for a state"""
	action_values = list(get_policy_actions(policy_lookup, state, player).values())
	if action_values:
		return np.max(action_values)
	return 0

def visualize_policy(policy_lookup, board, player):
	"""Display policy values of a board state"""
	state = board.flatten()
	# initialize board policy values to None
	board_policy_values = np.zeros((3, 3))
	for i in range(3):
		for j in range(3):
			board_policy_values[i][j] = None
	# lookup policy value for each action
	for action in board.get_open():
		board_policy_values[action[0], action[1]] = get_policy_value(
			policy_lookup, state, player, action)
	return board_policy_values
