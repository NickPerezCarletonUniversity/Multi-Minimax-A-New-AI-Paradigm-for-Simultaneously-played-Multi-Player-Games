import pygame
import time
import math
import copy
import random
import numpy as np
import scipy.stats
import gc

#some code taken from https://stackoverflow.com/questions/40363045/cant-draw-two-dimensional-array-pygame

clock = pygame.time.Clock()

pygame.init()

pygame.display.set_caption("Tron Bike Game")

test_board = 0


screen = pygame.display.set_mode((600, 600))

WHITE = (255, 255, 255)

GRAY = (50, 50, 50)

BLACK = (0, 0, 0)

RED = (255, 0, 0)

DARKRED = (100, 0, 0)

GREEN = (0, 255, 0)

DARKGREEN = (0, 100, 0)

BLUE = (0, 0, 255)

DARKBLUE = (0, 0, 100)

TURQOISE = (0, 255, 255)

DARKTURQOISE = (0, 100, 100)

MAGENTA = (255, 0, 255)

DARKMAGENTA = (100, 0, 100)

YELLOW = (255, 255, 0)

DARKYELLOW = (175, 175, 0)

LIGHTBROWN = (150, 125, 100)

BROWN = (100, 75, 50)


w = 59

deepest_search = 0

def bernoulli_confidence_interval(data):#https://stackoverflow.com/questions/10029588/python-implementation-of-the-wilson-score-interval
    n = len(data)

    if n == 0:
        return 0

    z = 1.96 #1.44 = 85%, 1.96 = 95%
    phat = sum(data) / n
    return round(100 * ((phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)), 2), round(100 * ((phat + z*z/(2*n) + z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)),2)



def mean_confidence_interval(data, confidence=0.95):#taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h, h


def are_players_eliminated(board, list_of_players):#players are numbered 0 to n-1, checking if players can be alive in their current position. returns a list with with playerIDs (player number) of eliminated players
	players_eliminated = []
	for i in range(len(list_of_players)):
		if list_of_players[i] == []:
			players_eliminated.append(i)
		elif list_of_players[i][0] < 0 or list_of_players[i][1] < 0 or list_of_players[i][0] >= len(board) or list_of_players[i][1] >= len(board[0]) or (board[list_of_players[i][0]][list_of_players[i][1]] != -1 and board[list_of_players[i][0]][list_of_players[i][1]] != i):
			players_eliminated.append(i)
		else:
			for j in range(len(list_of_players)):
					if i != j and list_of_players[i] == list_of_players[j]:
							players_eliminated.append(i)
	return players_eliminated

def valid_moves(board, player):#don't need to check for enemy players' current positions, since the AI_move function is called before players decide their moves
	if player == []:
		return []
	list_of_valid_moves = []
	if player[0] + 1 < len(board) and board[player[0]+1][player[1]] == -1:
		list_of_valid_moves.append((player[0] + 1, player[1]))
	if player[1] + 1 < len(board[0]) and board[player[0]][player[1]+1] == -1:
		list_of_valid_moves.append((player[0], player[1] + 1))
	if player[0] - 1 >= 0 and board[player[0]-1][player[1]] == -1:
		list_of_valid_moves.append((player[0] - 1, player[1]))
	if player[1] - 1 >= 0 and board[player[0]][player[1]-1] == -1:
		list_of_valid_moves.append((player[0], player[1] - 1))

	random.shuffle(list_of_valid_moves)

	return list_of_valid_moves

def valid_moves_no_shuffle(board, player):#don't need to check for enemy players' current positions, since the AI_move function is called before players decide their moves
	if player == []:
		return []
	list_of_valid_moves = []
	if player[0] + 1 < len(board) and board[player[0]+1][player[1]] == -1:
		list_of_valid_moves.append((player[0] + 1, player[1]))
	if player[1] + 1 < len(board[0]) and board[player[0]][player[1]+1] == -1:
		list_of_valid_moves.append((player[0], player[1] + 1))
	if player[0] - 1 >= 0 and board[player[0]-1][player[1]] == -1:
		list_of_valid_moves.append((player[0] - 1, player[1]))
	if player[1] - 1 >= 0 and board[player[0]][player[1]-1] == -1:
		list_of_valid_moves.append((player[0], player[1] - 1))

	return list_of_valid_moves

def valid_moves_directional(board, player):#don't need to check for enemy players' current positions, since the AI_move function is called before players decide their moves
	if player[0] == []:
		return []
	list_of_valid_moves = []
	if player[0][0] + 1 < len(board) and board[player[0][0]+1][player[0][1]] == -1:
		list_of_valid_moves.append(((player[0][0] + 1, player[0][1]), "right"))
	if player[0][1] + 1 < len(board[0]) and board[player[0][0]][player[0][1]+1] == -1:
		list_of_valid_moves.append(((player[0][0], player[0][1] + 1), "down"))
	if player[0][0] - 1 >= 0 and board[player[0][0]-1][player[0][1]] == -1:
		list_of_valid_moves.append(((player[0][0] - 1, player[0][1]), "left"))
	if player[0][1] - 1 >= 0 and board[player[0][0]][player[0][1]-1] == -1:
		list_of_valid_moves.append(((player[0][0], player[0][1] - 1), "up"))


	next_to_move = 0
	for i in range(len(list_of_valid_moves) - 1, -1, -1):
		if player[1] == list_of_valid_moves[i][1]:
			next_to_move = list_of_valid_moves.pop(i)
			return [next_to_move]#for super duper pruning!

	random.shuffle(list_of_valid_moves)

	return list_of_valid_moves

def valid_moves_directional_no_shuffle(board, player):#don't need to check for enemy players' current positions, since the AI_move function is called before players decide their moves
	if player[0] == []:
		return []
	list_of_valid_moves = []
	if player[0][0] + 1 < len(board) and board[player[0][0]+1][player[0][1]] == -1:
		list_of_valid_moves.append(((player[0][0] + 1, player[0][1]), "right"))
	if player[0][1] + 1 < len(board[0]) and board[player[0][0]][player[0][1]+1] == -1:
		list_of_valid_moves.append(((player[0][0], player[0][1] + 1), "down"))
	if player[0][0] - 1 >= 0 and board[player[0][0]-1][player[0][1]] == -1:
		list_of_valid_moves.append(((player[0][0] - 1, player[0][1]), "left"))
	if player[0][1] - 1 >= 0 and board[player[0][0]][player[0][1]-1] == -1:
		list_of_valid_moves.append(((player[0][0], player[0][1] - 1), "up"))


	next_to_move = 0
	for i in range(len(list_of_valid_moves) - 1, -1, -1):
		if player[1] == list_of_valid_moves[i][1]:
			next_to_move = list_of_valid_moves.pop(i)
			return [next_to_move]#for super duper pruning!

	return list_of_valid_moves

def nested_shuffle(valid_moves_2D, perspective_player):
	shuffled_and_flattened = []
	temp_list = list(range(len(valid_moves_2D)))
	del temp_list[perspective_player]
	for i in temp_list:
		for j in valid_moves_2D[i]:
			shuffled_and_flattened.append([j,i])

	random.shuffle(shuffled_and_flattened)

	return shuffled_and_flattened


def AI_move(player_number, current_player_position, board):#fills in a tile where the player is currently positioned and then returns the modified board
	board[current_player_position[0]][current_player_position[1]] = player_number
	return board

def console_print_board(board, list_of_players):#prints the board to console
	copy_of_board = copy.deepcopy(board)
	new_copy_of_board = []
	
	for j in range(len(copy_of_board[0])):
		temp = []
		for i in range(len(copy_of_board)):
			temp.append(copy_of_board[i][j])
		new_copy_of_board.append(copy.deepcopy(temp))

	copy_of_board = new_copy_of_board

	for i in range(len(copy_of_board)):
		for j in range(len(copy_of_board[0])):
			if copy_of_board[i][j] == -1:
				copy_of_board[i][j] = "-"
			elif copy_of_board[i][j] == 0:
				copy_of_board[i][j] = "g"
			elif copy_of_board[i][j] == 1:
				copy_of_board[i][j] = "b"
			elif copy_of_board[i][j] == 2:
				copy_of_board[i][j] = "r"
			elif copy_of_board[i][j] == 3:
				copy_of_board[i][j] = "d"
			elif copy_of_board[i][j] == 4:
				copy_of_board[i][j] = "t"
			elif copy_of_board[i][j] == 5:
				copy_of_board[i][j] = "m"
			elif copy_of_board[i][j] == 6:
				copy_of_board[i][j] = "y"
			elif copy_of_board[i][j] == 7:
				copy_of_board[i][j] = "l"
			if (j,i) == list_of_players[0]:
				copy_of_board[i][j] = "G"
			elif (j,i) == list_of_players[1]:
				copy_of_board[i][j] = "B"
			elif len(list_of_players) >= 3 and (j,i) == list_of_players[2]:
				copy_of_board[i][j] = "R"
			elif len(list_of_players) >= 4 and (j,i) == list_of_players[3]:
				copy_of_board[i][j] = "D"
			elif len(list_of_players) >= 5 and (j,i) == list_of_players[4]:
				copy_of_board[i][j] = "T"
			elif len(list_of_players) >= 6 and (j,i) == list_of_players[5]:
				copy_of_board[i][j] = "M"
			elif len(list_of_players) >= 7 and (j,i) == list_of_players[6]:
				copy_of_board[i][j] = "Y"
			elif len(list_of_players) >= 8 and (j,i) == list_of_players[7]:
				copy_of_board[i][j] = "L"

	str_to_print = ""
	for i in copy_of_board:
		str_to_print = str_to_print + "".join(map(str, i)) + "\n"
	print(str_to_print)

def best_reply_helper_directional_max(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth


	maximum_evaluation = -math.inf
	random.shuffle(list_of_valid_moves[perspective_player])
	for i in list_of_valid_moves[perspective_player]:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[perspective_player] = i
		temp = best_reply_helper_directional_min(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player)
		maximum_evaluation = max(maximum_evaluation, temp)
		alpha = max(alpha, maximum_evaluation)
		if beta <= alpha:
			break
	return maximum_evaluation


def best_reply_helper_directional_min(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	minimum_evaluation = math.inf
	for i in nested_shuffle(list_of_valid_moves, perspective_player):
		player_number = i[1]
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[player_number] = i[0]
		temp = best_reply_helper_directional_max(AI_move(player_number, copy_of_list_of_players[player_number][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player)
		minimum_evaluation = min(minimum_evaluation, temp)
		beta = min(beta, minimum_evaluation)
		if beta <= alpha:
			break
	return minimum_evaluation


def best_reply_directional(copy_of_board, max_depth, list_of_players, perspective_player):

	copy_of_list_of_players = copy.deepcopy(list_of_players)
	for i in range(len(list_of_players)):
		copy_of_list_of_players[i] = (copy_of_list_of_players[i], "")

	move_to_make = []


	maximum_evaluation = -math.inf
	for i in valid_moves_directional(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = best_reply_helper_directional_min(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, perspective_player)

		if temp == math.inf:
			return i[0]

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i[0]

	return move_to_make

def best_reply_helper_max(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	maximum_evaluation = -math.inf
	random.shuffle(list_of_valid_moves[perspective_player])
	for i in list_of_valid_moves[perspective_player]:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[perspective_player] = i
		temp = best_reply_helper_min(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player)
		maximum_evaluation = max(maximum_evaluation, temp)
		alpha = max(alpha, maximum_evaluation)
		if beta <= alpha:
			break
	return maximum_evaluation

def best_reply_helper_min(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	minimum_evaluation = math.inf
	for i in nested_shuffle(list_of_valid_moves, perspective_player):
		player_number = i[1]
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[player_number] = i[0]
		temp = best_reply_helper_max(AI_move(player_number, copy_of_list_of_players[player_number], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player)
		minimum_evaluation = min(minimum_evaluation, temp)
		beta = min(beta, minimum_evaluation)
		if beta <= alpha:
			break
	return minimum_evaluation

def best_reply(copy_of_board, max_depth, list_of_players, perspective_player):#player 0 will be human player 1 will be AI, to start
	possible_moves_to_make = []
	copy_of_list_of_players = copy.deepcopy(list_of_players)

	move_to_make = []

	maximum_evaluation = -math.inf
	for i in valid_moves(copy_of_board, list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = best_reply_helper_min(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, perspective_player)

		if temp == math.inf:
			return i

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i

	return move_to_make

def multi_minimax_directional_helper_helper_max(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player, enemy_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))

	if current_depth == max_depth or list_of_valid_moves[0] == []:
		return current_depth

	if list_of_valid_moves[1] == []:
		return math.inf


	maximum_evaluation = -math.inf
	random.shuffle(list_of_valid_moves[0])
	for i in list_of_valid_moves[0]:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[0] = i
		temp = multi_minimax_directional_helper_helper_min(AI_move(perspective_player, copy_of_list_of_players[0][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player, enemy_player)
		maximum_evaluation = max(maximum_evaluation, temp)
		alpha = max(alpha, maximum_evaluation)
		if beta <= alpha:
			break
	return maximum_evaluation


def multi_minimax_directional_helper_helper_min(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player, enemy_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))

	if current_depth == max_depth or list_of_valid_moves[0] == []:
		return current_depth

	if list_of_valid_moves[1] == []:
		return math.inf

	minimum_evaluation = math.inf
	random.shuffle(list_of_valid_moves[1])
	for i in list_of_valid_moves[1]:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[1] = i
		temp = multi_minimax_directional_helper_helper_max(AI_move(enemy_player, copy_of_list_of_players[1][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, perspective_player, enemy_player)
		minimum_evaluation = min(minimum_evaluation, temp)
		beta = min(beta, minimum_evaluation)
		if beta <= alpha:
			break
	return minimum_evaluation

def multi_minimax_directional_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0


	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth


	minimum_evaluation = math.inf
	for i in nested_shuffle(list_of_valid_moves, perspective_player):
		player_number = i[1]
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[player_number] = i[0]
		temp = multi_minimax_directional_helper_helper_max(AI_move(player_number, copy_of_list_of_players[player_number][0], copy.deepcopy(copy_of_board)), max_depth, 2, alpha, beta, copy.deepcopy([copy_of_list_of_players[perspective_player], copy_of_list_of_players[player_number]]), perspective_player, player_number)
		minimum_evaluation = min(minimum_evaluation, temp)
		beta = min(beta, minimum_evaluation)
		if beta <= alpha:
			break
	return minimum_evaluation


def multi_minimax_directional(copy_of_board, max_depth, list_of_players, perspective_player):

	copy_of_list_of_players = copy.deepcopy(list_of_players)
	for i in range(len(list_of_players)):
		copy_of_list_of_players[i] = (copy_of_list_of_players[i], "")

	move_to_make = []

	maximum_evaluation = -math.inf
	for i in valid_moves_directional(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = multi_minimax_directional_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, perspective_player)

		if temp == math.inf:
			return i[0]

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i[0]

	return move_to_make


def multi_minimax_helper_helper(copy_of_board, max_depth, current_depth, maximizing, alpha, beta, list_of_players, perspective_player, enemy_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))

	if current_depth == max_depth or list_of_valid_moves[0] == []:
		return current_depth

	if list_of_valid_moves[1] == []:
		return math.inf


	if maximizing == True:
		maximum_evaluation = -math.inf
		random.shuffle(list_of_valid_moves[0])
		for i in list_of_valid_moves[0]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[0] = i
			temp = multi_minimax_helper_helper(AI_move(perspective_player, copy_of_list_of_players[0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, False, alpha, beta, copy_of_list_of_players, perspective_player, enemy_player)
			maximum_evaluation = max(maximum_evaluation, temp)
			alpha = max(alpha, maximum_evaluation)
			if beta <= alpha:
				break
		return maximum_evaluation

	else:
		minimum_evaluation = math.inf
		random.shuffle(list_of_valid_moves[1])
		for i in list_of_valid_moves[1]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[1] = i
			temp = multi_minimax_helper_helper(AI_move(enemy_player, copy_of_list_of_players[1], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, True, alpha, beta, copy_of_list_of_players, perspective_player, enemy_player)
			minimum_evaluation = min(minimum_evaluation, temp)
			beta = min(beta, minimum_evaluation)
			if beta <= alpha:
				break
		return minimum_evaluation

def multi_minimax_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, perspective_player):
	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []#list_of_valid_moves is a 2d array

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	minimum_evaluation = math.inf
	for i in nested_shuffle(list_of_valid_moves, perspective_player):
		player_number = i[1]
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[player_number] = i[0]
		temp = multi_minimax_helper_helper(AI_move(player_number, copy_of_list_of_players[player_number], copy.deepcopy(copy_of_board)), max_depth, 2, True, alpha, beta, copy.deepcopy([copy_of_list_of_players[perspective_player], copy_of_list_of_players[player_number]]), perspective_player, player_number)
		minimum_evaluation = min(minimum_evaluation, temp)
		beta = min(beta, minimum_evaluation)
		if beta <= alpha:
			break
	return minimum_evaluation

def multi_minimax(copy_of_board, max_depth, list_of_players, perspective_player):

	copy_of_list_of_players = copy.deepcopy(list_of_players)

	move_to_make = []

	maximum_evaluation = -math.inf
	for i in valid_moves(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = multi_minimax_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, perspective_player)

		if temp == math.inf:
			return i

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i

	return move_to_make


def paranoid_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, move_order, perspective_player, current_player):
	actual_current_player = move_order[current_player]

	if actual_current_player != perspective_player and list_of_players[actual_current_player] == []:
		return paranoid_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	if perspective_player == actual_current_player:
		random.shuffle(list_of_valid_moves[perspective_player])
		maximum_evaluation = -math.inf
		for i in list_of_valid_moves[perspective_player]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[perspective_player] = i
			temp = paranoid_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))
			maximum_evaluation = max(maximum_evaluation, temp)
			alpha = max(alpha, maximum_evaluation)
			if beta <= alpha:
				break
		return maximum_evaluation

	else:
		if list_of_valid_moves[actual_current_player] == []:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[actual_current_player] = []
			return paranoid_helper(copy_of_board, max_depth, current_depth, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))
		minimum_evaluation = math.inf
		random.shuffle(list_of_valid_moves[actual_current_player])
		for i in list_of_valid_moves[actual_current_player]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[actual_current_player] = i
			temp = paranoid_helper(AI_move(actual_current_player, copy_of_list_of_players[actual_current_player], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))
			minimum_evaluation = min(minimum_evaluation, temp)
			beta = min(beta, minimum_evaluation)
			if beta <= alpha:
				break
		return minimum_evaluation

def paranoid(copy_of_board, max_depth, list_of_players, perspective_player):
	copy_of_list_of_players = copy.deepcopy(list_of_players)

	temp_list = list(range(len(list_of_players)))

	random.shuffle(temp_list)

	current_player = (temp_list.index(perspective_player) + 1) % len(list_of_players)

	move_to_make = []

	maximum_evaluation = -math.inf
	for i in valid_moves(copy_of_board, list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = paranoid_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, temp_list, perspective_player, current_player)

		if temp == math.inf:
			return i

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i
		elif maximum_evaluation == temp:
			move_to_make = i

	return move_to_make



def paranoid_directional_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, move_order, perspective_player, current_player):
	actual_current_player = move_order[current_player]

	if actual_current_player != perspective_player and list_of_players[actual_current_player][0] == []:
		return paranoid_directional_helper(copy_of_board, max_depth, current_depth, alpha, beta, list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return current_depth

	if winning_board:
		return math.inf

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return current_depth

	if actual_current_player == perspective_player:
		maximum_evaluation = -math.inf
		random.shuffle(list_of_valid_moves[perspective_player])
		for i in list_of_valid_moves[perspective_player]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[perspective_player] = i
			temp = paranoid_directional_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))
			maximum_evaluation = max(maximum_evaluation, temp)
			alpha = max(alpha, maximum_evaluation)
			if beta <= alpha:
				break
		return maximum_evaluation


	else:
		if list_of_valid_moves[actual_current_player] == []:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[actual_current_player] = ([], "")
			return paranoid_directional_helper(copy_of_board, max_depth, current_depth, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

		minimum_evaluation = math.inf
		random.shuffle(list_of_valid_moves[actual_current_player])
		for i in list_of_valid_moves[actual_current_player]:
			copy_of_list_of_players = copy.deepcopy(list_of_players)
			copy_of_list_of_players[actual_current_player] = i
			temp = paranoid_directional_helper(AI_move(actual_current_player, copy_of_list_of_players[actual_current_player][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, alpha, beta, copy_of_list_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))
			minimum_evaluation = min(minimum_evaluation, temp)
			beta = min(beta, minimum_evaluation)
			if beta <= alpha:
				break
		return minimum_evaluation

def paranoid_directional(copy_of_board, max_depth, list_of_players, perspective_player):
	copy_of_list_of_players = copy.deepcopy(list_of_players)

	for i in range(len(list_of_players)):
		copy_of_list_of_players[i] = (copy_of_list_of_players[i], "")


	temp_list = list(range(len(list_of_players)))

	random.shuffle(temp_list)

	current_player = (temp_list.index(perspective_player) + 1) % len(list_of_players)

	move_to_make = []

	maximum_evaluation = -math.inf
	for i in valid_moves_directional(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp = paranoid_directional_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, 1, maximum_evaluation, math.inf, copy_of_list_of_players, temp_list, perspective_player, current_player)

		if temp == math.inf:
			return i[0]

		if maximum_evaluation < temp:
			maximum_evaluation = temp
			move_to_make = i[0]
		elif maximum_evaluation == temp:
			move_to_make = i[0]

	return move_to_make



def maxn_helper(copy_of_board, max_depth, current_depth, list_of_players, eval_of_players, move_order, perspective_player, current_player):
	actual_current_player = move_order[current_player]

	if actual_current_player != perspective_player and list_of_players[actual_current_player] == []:
		return maxn_helper(copy_of_board, max_depth, current_depth, list_of_players, eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))	

	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return eval_of_players

	if winning_board:
		copy_of_eval_of_players = eval_of_players.copy()
		copy_of_eval_of_players[actual_current_player] = math.inf
		return copy_of_eval_of_players

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return eval_of_players

	if list_of_valid_moves[actual_current_player] == []:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[actual_current_player] = []
		return maxn_helper(copy_of_board, max_depth, current_depth, copy_of_list_of_players, eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

	maximum_evaluation = [0] * len(list_of_players)
	random.shuffle(list_of_valid_moves[actual_current_player])
	for i in list_of_valid_moves[actual_current_player]:

		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[actual_current_player] = i
		copy_of_eval_of_players = eval_of_players.copy()
		copy_of_eval_of_players[actual_current_player] = copy_of_eval_of_players[actual_current_player] + 1
		temp = maxn_helper(AI_move(actual_current_player, copy_of_list_of_players[actual_current_player], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, copy_of_list_of_players, copy_of_eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

		eval_of_temp = temp[actual_current_player]

		if eval_of_temp == math.inf:
			return temp

		if maximum_evaluation[actual_current_player] < eval_of_temp:
			maximum_evaluation = temp
		elif maximum_evaluation[actual_current_player] == eval_of_temp:
			if sum(temp) < sum(maximum_evaluation):
				maximum_evaluation = temp

	return maximum_evaluation


def maxn(copy_of_board, max_depth, list_of_players, perspective_player):
	copy_of_list_of_players = copy.deepcopy(list_of_players)

	temp_list = list(range(len(list_of_players)))

	random.shuffle(temp_list)

	current_player = (temp_list.index(perspective_player) + 1) % len(list_of_players)

	move_to_make = []

	maximum_evaluation = [0] * len(list_of_players)
	for i in valid_moves(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp_eval_list = [0] * len(list_of_players)
		temp_eval_list[perspective_player] = 1
		temp = maxn_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player], copy.deepcopy(copy_of_board)), max_depth, 1, copy_of_list_of_players, temp_eval_list, temp_list, perspective_player, current_player)
		eval_of_temp = temp[perspective_player]

		if eval_of_temp == math.inf:
			return i

		if maximum_evaluation[perspective_player] < eval_of_temp:
			maximum_evaluation = temp
			move_to_make = i
		elif maximum_evaluation[perspective_player] == eval_of_temp:
			if sum(temp) < sum(maximum_evaluation):
				move_to_make = i
				maximum_evaluation = temp

	return move_to_make

def maxn_directional_helper(copy_of_board, max_depth, current_depth, list_of_players, eval_of_players, move_order, perspective_player, current_player):
	actual_current_player = move_order[current_player]

	if actual_current_player != perspective_player and list_of_players[actual_current_player][0] == []:
		return maxn_directional_helper(copy_of_board, max_depth, current_depth, list_of_players, eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))	

	global deepest_search
	deepest_search = max(deepest_search, current_depth)

	list_of_valid_moves = []

	winning_board = True

	no_valid_moves = True

	temp_index = 0

	for i in list_of_players:
		list_of_valid_moves.append(valid_moves_directional_no_shuffle(copy_of_board, i))
		if list_of_valid_moves[temp_index] != [] and temp_index != perspective_player:
			 winning_board = False
			 no_valid_moves = False
		temp_index = temp_index + 1

	if list_of_valid_moves[perspective_player] != []:
		no_valid_moves = False
		
	if no_valid_moves == True:
		return eval_of_players

	if winning_board:
		copy_of_eval_of_players = eval_of_players.copy()
		copy_of_eval_of_players[actual_current_player] = math.inf
		return copy_of_eval_of_players

	if current_depth == max_depth or list_of_valid_moves[perspective_player] == []:
		return eval_of_players

	if list_of_valid_moves[actual_current_player] == []:
		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[actual_current_player] = ([], "")
		return maxn_directional_helper(copy_of_board, max_depth, current_depth, copy_of_list_of_players, eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

	maximum_evaluation = [0] * len(list_of_players)
	random.shuffle(list_of_valid_moves[actual_current_player])
	for i in list_of_valid_moves[actual_current_player]:

		copy_of_list_of_players = copy.deepcopy(list_of_players)
		copy_of_list_of_players[actual_current_player] = i
		copy_of_eval_of_players = eval_of_players.copy()
		copy_of_eval_of_players[actual_current_player] = copy_of_eval_of_players[actual_current_player] + 1
		temp = maxn_directional_helper(AI_move(actual_current_player, copy_of_list_of_players[actual_current_player][0], copy.deepcopy(copy_of_board)), max_depth, current_depth + 1, copy_of_list_of_players, copy_of_eval_of_players, move_order, perspective_player, (current_player + 1) % len(list_of_players))

		eval_of_temp = temp[actual_current_player]

		if eval_of_temp == math.inf:
			return temp

		if maximum_evaluation[actual_current_player] < eval_of_temp:
			maximum_evaluation = temp
		elif maximum_evaluation[actual_current_player] == eval_of_temp:
			if sum(temp) < sum(maximum_evaluation):
				maximum_evaluation = temp

	return maximum_evaluation


def maxn_directional(copy_of_board, max_depth, list_of_players, perspective_player):
	copy_of_list_of_players = copy.deepcopy(list_of_players)

	for i in range(len(list_of_players)):
		copy_of_list_of_players[i] = (copy_of_list_of_players[i], "")

	temp_list = list(range(len(list_of_players)))

	random.shuffle(temp_list)

	current_player = (temp_list.index(perspective_player) + 1) % len(list_of_players)

	move_to_make = []

	maximum_evaluation = [0] * len(list_of_players)
	for i in valid_moves_directional(copy_of_board, copy_of_list_of_players[perspective_player]):
		copy_of_list_of_players[perspective_player] = i
		temp_eval_list = [0] * len(list_of_players)
		temp_eval_list[perspective_player] = 1
		temp = maxn_directional_helper(AI_move(perspective_player, copy_of_list_of_players[perspective_player][0], copy.deepcopy(copy_of_board)), max_depth, 1, copy_of_list_of_players, temp_eval_list, temp_list, perspective_player, current_player)
		eval_of_temp = temp[perspective_player]

		if eval_of_temp == math.inf:
			return i[0]

		if maximum_evaluation[perspective_player] < eval_of_temp:
			maximum_evaluation = temp
			move_to_make = i[0]
		elif maximum_evaluation[perspective_player] == eval_of_temp:
			if sum(temp) < sum(maximum_evaluation):
				move_to_make = i[0]
				maximum_evaluation = temp

	return move_to_make



def random_player_move(player, board):
	list_of_valid_moves = valid_moves(board, player)
	if list_of_valid_moves == []:
		return []
	else:
		return list_of_valid_moves[0]

def initiate_players(number_of_players, width, height):
	players = []

	for i in range(number_of_players):
		players.append([])

	#20 by 20
	#temp_list = [[5,0],[14,0],[0,5],[0,14],[5,19],[14,19],[19,5],[19,14]]
	#16 by 16
	#temp_list = [[4,0],[11,0],[0,4],[0,11],[4,15],[11,15],[15,4],[15,11]]
	#12 by 12
	temp_list = [[3,0],[8,0],[0,3],[0,8],[3,11],[8,11],[11,3],[11,8]]
	#10 by 10
	#temp_list = [[2,0],[7,0],[0,2],[0,7],[2,9],[7,9],[9,2],[9,7]]
	random.shuffle(temp_list)
	for i in range(number_of_players):
		duplicate_position = True
		while duplicate_position == True:
			players[i] = temp_list[i]
			#players[i] = (random.randint(0, width-1), random.randint(0, height-1))
			duplicate_position = False
			for j in range(number_of_players):
				if players[i] == players[j] and j != i:
					duplicate_position = True
	return players

#some code taken from https://stackoverflow.com/questions/20748326/pygame-waiting-the-user-to-keypress-a-key
def player_move(player, board):
	board[player[0]][player[1]] = 0
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
				return (player[0] + 1, player[1])
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
				return (player[0] - 1, player[1])
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
				return (player[0], player[1] + 1)
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
				return (player[0], player[1] - 1)

def initialize_board(width, height):#-1 denotes an unused board tile
	board = []
	for i in range(width):
		new_row_to_add = []
		for j in range(height):
			new_row_to_add.append(-1)
		board.append(new_row_to_add)
	return board


def draw_board(board, players):# board is represented as a 2d array. Each column is a list. the board is oriented so that the top left corner of the screen is (0,0), and increases to the right and downwards
	w = (600 / max(len(board[0]), len(board))) - 1
	x = 0
	y = 0
	for column in range(len(board)):
		for row in range(len(board[0])):
			box = pygame.Rect(x, y, w, w)
			if (column, row) == players[0]:#-1 denotes an unused board tile, and any number x greater than or equal to 0 denotes one of the xth player's wall pieces they put up
				pygame.draw.rect(screen, GREEN, box)
			elif board[column][row] == 0:
				pygame.draw.rect(screen, DARKGREEN, box)
			elif (column, row) == players[1]:
				pygame.draw.rect(screen, BLUE, box)
			elif board[column][row] == 1:
				pygame.draw.rect(screen, DARKBLUE, box)
			elif len(players) >= 3 and (column, row) == players[2]:
				pygame.draw.rect(screen, RED, box)
			elif len(players) >= 3 and board[column][row] == 2:
				pygame.draw.rect(screen, DARKRED, box)
			elif len(players) >= 4 and (column, row) == players[3]:
				pygame.draw.rect(screen, GRAY, box)
			elif len(players) >= 4 and board[column][row] == 3:
				pygame.draw.rect(screen, BLACK, box)
			elif len(players) >= 5 and (column, row) == players[4]:
				pygame.draw.rect(screen, TURQOISE, box)
			elif len(players) >= 5 and board[column][row] == 4:
				pygame.draw.rect(screen, DARKTURQOISE, box)
			elif len(players) >= 6 and (column, row) == players[5]:
				pygame.draw.rect(screen, MAGENTA, box)
			elif len(players) >= 6 and board[column][row] == 5:
				pygame.draw.rect(screen, DARKMAGENTA, box)
			elif len(players) >= 7 and (column, row) == players[6]:
				pygame.draw.rect(screen, YELLOW, box)
			elif len(players) >= 7 and board[column][row] == 6:
				pygame.draw.rect(screen, DARKYELLOW, box)
			elif len(players) >= 8 and (column, row) == players[7]:
				pygame.draw.rect(screen, LIGHTBROWN, box)
			elif len(players) >= 8 and board[column][row] == 7:
				pygame.draw.rect(screen, BROWN, box)
			else:
				pygame.draw.rect(screen, WHITE, box)
			y = y + w + 1
		x = x + w + 1
		y = 0



running = True

board_width = 12
board_height = 12

number_of_players = 3

players = initiate_players(number_of_players, board_width, board_height)

board = initialize_board(board_width, board_height)

number_of_tests = 5000

current_test_number = 0

number_of_wins = 0

number_of_ties = 0

number_of_wins_list = []

number_of_wins_and_ties_list = []

max_depth_list = []


while running:
	clock.tick(3)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
			running = False

	draw_board(board, players)
	pygame.display.flip()
	

	players_new_positions = copy.deepcopy(players)


	for i in range(len(players)):
		if players[i] != []:
			AI_move(i, players[i], board)


	for i in range(len(players)):
		
		if i == 0:
			iterative_depth = 0
			start = time.time()
			deepest_search = 0
			actual_deepest_search = 0
			gc.collect()

			while True:
				iterative_depth = iterative_depth + 1
				temp = multi_minimax_directional(board, iterative_depth, players, i)
				if (time.time() - start) > 0.3:
					break
				else:
					players_new_positions[i] = temp
					actual_deepest_search = deepest_search

			max_depth_list.append(actual_deepest_search)
			print("player " + str(i) + " went to a depth of " + str(actual_deepest_search))

		else:

			players_new_positions[i] = random_player_move(players[i], board)




	players = players_new_positions

	if are_players_eliminated(board, players) != []:
		for i in are_players_eliminated(board, players):
			players[i] = []


	console_print_board(board, players)


	if are_players_eliminated(board, players) == [1, 2]:
		number_of_wins = number_of_wins + 1
		players[0] = []
		number_of_wins_list.append(1)
		number_of_wins_and_ties_list.append(1)
	elif are_players_eliminated(board, players) != [] and are_players_eliminated(board, players)[0] == 0:
		if are_players_eliminated(board, players) == [0, 1, 2]:
			number_of_ties = number_of_ties + 1
			number_of_wins_and_ties_list.append(1)
			number_of_wins_list.append(0)
		else:
			players = [[],[],[]]
			number_of_wins_and_ties_list.append(0)
			number_of_wins_list.append(0)

	game_over = True
	for i in players:
		if i != []:
			game_over = False
			

	if game_over == True:
		board = initialize_board(board_width, board_height)
		players = initiate_players(number_of_players, board_width, board_height)
		current_test_number = current_test_number + 1
		if current_test_number == number_of_tests:
			break

	if current_test_number != 0:
		print("so far it has won " + str(number_of_wins/current_test_number) + " of the time.")
		print("so far it has tied " + str(number_of_ties/current_test_number) + " of the time.")
		print("on test number: " + str(current_test_number) + " out of " + str(number_of_tests) + " tests.")

print("")
print("out of " + str(number_of_tests) + " games...")
print("number of ties: " + str (number_of_ties))
print("number_of_wins: " + str(number_of_wins))

print("length of wins list: " + str(len(number_of_wins_list)))
print("length of wins and ties list: " + str(len(number_of_wins_and_ties_list)))
print("confidence interval for wins:")
print(bernoulli_confidence_interval(number_of_wins_list))
print("confidence interval for wins and ties:")
print(bernoulli_confidence_interval(number_of_wins_and_ties_list))
print("confidence interval for maximum depth:")
print(mean_confidence_interval(max_depth_list, 0.95))



pygame.quit()