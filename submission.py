import logic
import random
from AbstractPlayers import *
import time
import math

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


grades_dict = {
    1: 110,
    2: 100,
    4: 95,
    8: 90,
    16: 85,
    32: 80,
    64: 75,
    128: 70,
    256: 60,
    512: 55,
    1024: 50,
    2048: 45
}


def heuristic_val(new_board, score):
    val = zero_heuristic(new_board)  # number between 0 to 100
    val += max_tile_heuristic(new_board)  # number between 0 to 100
    val += 2 * monotonicity_heuristic(new_board)  # number between 0 to 100
    val += 1.5 * smooth_heuristic(new_board)
    val += 1000 * (score / max(max(new_board, key=lambda x: max(x))))
    return val


def smooth_heuristic(new_board):
    grades_list = []
    for i in range(4):
        for j in range(3):
            if not (new_board[i][j] == 0) and not (new_board[i][j + 1] == 0):
                diff = max(new_board[i][j] / new_board[i][j + 1], new_board[i][j + 1] / new_board[i][j])
                if diff <= 2048:
                    grades_list.append(grades_dict[diff])
                else:
                    grades_list.append(0)
            else:
                grades_list.append(0)
    for j in range(4):
        for i in range(3):
            if not (new_board[i][j] == 0) and not (new_board[i + 1][j] == 0):
                diff = max(new_board[i][j] / new_board[i + 1][j], new_board[i + 1][j] / new_board[i][j])
                if diff <= 2048:
                    grades_list.append(grades_dict[diff])
                else:
                    grades_list.append(0)
            else:
                grades_list.append(0)
    return sum(grades_list) / len(grades_list)


def zero_heuristic(new_board):
    zero_num = 0
    for i in range(len(new_board)):
        for j in range(len(new_board[0])):
            if new_board[i][j] == 0:
                zero_num += 1
    return min(100, int((zero_num * 100) / 12))


def max_tile_heuristic(board):
    max_value = max(max(board, key=lambda x: max(x)))
    if board[0][0] == max_value or board[0][3] == max_value:
        return 100
    if board[3][0] == max_value or board[3][3] == max_value:
        return 50
    return 0


def monotonicity_heuristic(new_board):
    best = -1
    for _ in range(4):
        current_row = 0
        current_col = 0
        for i in range(len(new_board)):
            for j in range(len(new_board[0]) - 1):
                if new_board[i][j] >= new_board[i][j + 1]:
                    current_row += 1

        for j in range(len(new_board[0])):
            for i in range(len(new_board) - 1):
                if new_board[i][j] >= new_board[i + 1][j]:
                    current_col += 1
        max_curr = max(current_row, current_col)
        if max_curr > best:
            best = max_curr
        new_board = logic.transpose(logic.reverse(new_board))
    return (best * 100) / 12


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = heuristic_val(new_board, score)
        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_move(self, board, time_limit) -> Move:
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        final_depth = 0
        last_move = Move.UP
        try:
            while self.getRemainingTime() > 0.05 and depth <= 23:
                depth += 1
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.minimax(new_board, depth, 'Index', score)
                last_move = max(optional_moves_score, key=optional_moves_score.get)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            # print(final_depth)
            return last_move

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def minimax(self, curr_board, depth, player_type, curr_score):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.minimax(new_board, depth - 1, 'Index', curr_score + score)
                    max_eval = max(curr_val, max_eval)
                if self.getRemainingTime() < 0.05:
                    raise TimeoutError
            return max_eval
        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = 2
                        curr_index_val = self.minimax(curr_board, depth - 1, 'Move', curr_score)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
            return min_eval


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        last_move = (0, 0)
        try:
            while self.getRemainingTime() > 0.05 and depth <= 23:
                depth += 1
                optional_moves_score = {}
                for i in range(len(board)):
                    for j in range(len(board[0])):
                        if board[i][j] == 0:
                            board[i][j] = 2
                            curr_index_val = self.minimax(board, depth - 1, 'Move', 0)
                            optional_moves_score[(i, j)] = curr_index_val
                            board[i][j] = 0
                last_move = min(optional_moves_score, key=optional_moves_score.get)
                # print(last_move)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            # print(final_depth)
            #print(last_move)
            return last_move[0], last_move[1]

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def minimax(self, curr_board, depth, player_type, curr_score):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.minimax(new_board, depth - 1, 'Index', curr_score+score)
                    max_eval = max(curr_val, max_eval)
                if self.getRemainingTime() < 0.05:
                    raise TimeoutError
            return max_eval
        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = 2
                        curr_index_val = self.minimax(curr_board, depth - 1, 'Move', curr_score)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
            return min_eval


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_move(self, board, time_limit) -> Move:
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        final_depth = 0
        last_move = Move.UP
        a = -math.inf
        b = math.inf
        try:
            while self.getRemainingTime() > 0.05 and depth <= 23:
                depth += 1
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.ABminimax(new_board, depth, 'Index', a, b, score)
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
                last_move = max(optional_moves_score, key=optional_moves_score.get)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            # print(final_depth)
            return last_move

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def ABminimax(self, curr_board, depth, player_type, a, b, curr_score):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.ABminimax(new_board, depth - 1, 'Index', a, b, curr_score + score)
                    max_eval = max(curr_val, max_eval)
                    a = max(max_eval, a)
                    if max_eval >= b:
                        return math.inf
                if self.getRemainingTime() < 0.05:
                    raise TimeoutError
            return max_eval
        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = 2
                        curr_index_val = self.ABminimax(curr_board, depth - 1, 'Move', a, b, curr_score)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                        b = min(b, min_eval)
                        if min_eval <= a:
                            return -math.inf
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
            return min_eval


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        final_depth = 0
        last_move = Move.UP
        try:
            while self.getRemainingTime() > 0.1 and depth <= 23:
                depth += 1
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.expectimax(new_board, score, depth, 'Chance', 0)
                last_move = max(optional_moves_score, key=optional_moves_score.get)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            #print(final_depth)
            return last_move

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def expectimax(self, curr_board, curr_score, depth, player_type, index_value):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Chance':
            if self.getRemainingTime() < 0.1:
                raise TimeoutError
            value_2 = self.expectimax(curr_board, curr_score, depth-1, 'Index', 2)
            value_4 = self.expectimax(curr_board, curr_score, depth-1, 'Index', 4)
            return 0.9*value_2 + 0.1*value_4

        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.expectimax(new_board, curr_score + score, depth-1, 'Chance', 0)
                    max_eval = max(curr_val, max_eval)
                if self.getRemainingTime() < 0.1:
                    raise TimeoutError
            return max_eval

        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = index_value
                        curr_index_val = self.expectimax(curr_board, curr_score, depth-1, 'Move', 0)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                    if self.getRemainingTime() < 0.1:
                        raise TimeoutError
            return min_eval


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        final_depth = 0
        last_move = (0, 0)
        try:
            while self.getRemainingTime() > 0.1 and depth <= 23:
                depth += 1
                score_list = []
                optional_moves_score = {}
                for i in range(len(board)):
                    for j in range(len(board[0])):
                        if board[i][j] == 0:
                            board[i][j] = value
                            curr_index_val = self.expectimax(board, 0, depth - 1, 'Move', 0)
                            optional_moves_score[(i, j)] = curr_index_val
                            board[i][j] = 0
                            last_move = min(optional_moves_score, key=optional_moves_score.get)
                #print(last_move)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            #print(final_depth)
            return last_move[0], last_move[1]

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def expectimax(self, curr_board, curr_score, depth, player_type, index_value):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Chance':
            if self.getRemainingTime() < 0.1:
                raise TimeoutError
            value_2 = self.expectimax(curr_board, curr_score, depth-1, 'Index', 2)
            value_4 = self.expectimax(curr_board, curr_score, depth-1, 'Index', 4)
            return 0.9*value_2 + 0.1*value_4

        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.expectimax(new_board, curr_score + score, depth-1, 'Chance', 0)
                    max_eval = max(curr_val, max_eval)
                if self.getRemainingTime() < 0.1:
                    raise TimeoutError
            return max_eval

        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = index_value
                        curr_index_val = self.expectimax(curr_board, curr_score, depth-1, 'Move', 0)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                    if self.getRemainingTime() < 0.1:
                        raise TimeoutError
            return min_eval

# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.minimax_time_limit = 0
        self.minimax_start_time = 0

    def get_move(self, board, time_limit) -> Move:
        self.minimax_start_time = time.time()
        self.minimax_time_limit = time_limit
        depth = 0
        final_depth = 0
        last_move = Move.UP
        a = -math.inf
        b = math.inf
        try:
            while self.getRemainingTime() > 0.05 and depth <= 23:
                depth += 1
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.ABminimax(new_board, depth, 'Index', a, b, score)
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
                last_move = max(optional_moves_score, key=optional_moves_score.get)
                final_depth = depth
            raise TimeoutError
        except TimeoutError:
            # print(final_depth)
            return last_move

    # TODO: add here helper functions in class, if needed
    def getRemainingTime(self):
        return self.minimax_time_limit - (time.time() - self.minimax_start_time)

    def ABminimax(self, curr_board, depth, player_type, a, b, curr_score):
        if depth == 0 or logic.game_state(curr_board) == 'lose':
            return heuristic_val(curr_board, curr_score)
        if player_type == 'Move':
            max_eval = -math.inf
            for move in Move:
                new_board, done, score = commands[move](curr_board)
                if done:
                    curr_val = self.ABminimax(new_board, depth - 1, 'Index', a, b, curr_score + score)
                    max_eval = max(curr_val, max_eval)
                    a = max(max_eval, a)
                    if max_eval >= b:
                        return math.inf
                if self.getRemainingTime() < 0.05:
                    raise TimeoutError
            return max_eval
        else:
            min_eval = math.inf
            for i in range(len(curr_board)):
                for j in range(len(curr_board[0])):
                    if curr_board[i][j] == 0:
                        curr_board[i][j] = 2
                        curr_index_val = self.ABminimax(curr_board, depth - 1, 'Move', a, b, curr_score)
                        curr_board[i][j] = 0
                        min_eval = min(min_eval, curr_index_val)
                        b = min(b, min_eval)
                        if min_eval <= a:
                            return -math.inf
                    if self.getRemainingTime() < 0.05:
                        raise TimeoutError
            return min_eval
