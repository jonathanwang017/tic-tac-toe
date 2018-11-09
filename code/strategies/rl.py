import numpy as np

from base.board import *

discount = 1
learning_rate = 0.1

def learn_policy(policy_lookup):
	"""Play games to learn optimal policy"""
	# epsilon = 1
	for i in range(10000):
		board = Board()
		turn = 1
		visualize_policy(policy_lookup, board, turn)
		while not board.check_end():
			state = board.flatten()
			# if np.random.random() > epsilon:
			# 	action = exploit_policy(policy_lookup, board, turn)
			# else:
			action = explore_policy(board)

			board.add_piece(turn, action[0], action[1])

			next_state = board.flatten()
			reward = get_reward(board)

			update_policy_values(
				policy_lookup, board, state, turn, action, next_state, reward)

			turn = 3 - turn

		# epsilon = max(0, epsilon - 0.05)
	
def explore_policy(board):
	"""Select random move"""
	open_spaces = board.get_open()
	rand_ind = np.random.randint(len(open_spaces))
	return open_spaces[rand_ind]
	
		
def exploit_policy(policy_lookup, board, player):
	"""Select moves based on policy and evaluate rewards"""
	state = board.flatten()
	best_value = None
	best_action = None
	for action in board.get_open():
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
		if board.get_winner() == 1:
			return 1
		elif board.get_winner() == 2:
			return -1
		else:
			return 0
	else:
		return 0

def update_policy_values(policy_lookup, board, state, player, action, next_state, reward):
	"""Update policy values with Bellman equation"""
	if board.check_end():
		expected = reward
	else:
		if player == 1:
			expected = reward + discount * min_value(policy_lookup, next_state, 2)
		elif player == 2:
			expected = reward + discount * max_value(policy_lookup, next_state, 1)
	policy_value = get_policy_value(policy_lookup, state, player, action)
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
	print(board_policy_values)
