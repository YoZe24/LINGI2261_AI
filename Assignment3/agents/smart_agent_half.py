from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
import time, math, random
from seega import SeegaAction
import random
import numpy as np

edge_middle = [(0, 2), (2, 4), (4, 2), (2, 0)]
around_center = [(1, 2), (2, 3), (3, 2), (2, 1)]
corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
next_to_corners = [(0, 1), (1, 0),  # next to top left
                   (0, 3), (1, 4),  # next to top right
                   (3, 4), (4, 3),  # next to bottom right
                   (3, 0), (4, 1)]  # next to bottom left

next_to_center = [[(0, 2), (1, 2)],  # cet ordre est préférable"
                  [(2, 3), (2, 4)],
                  [(3, 2), (4, 2)],
                  [(2, 0), (2, 1)]]

to_not_avoid = [(0, 2), (1, 1), (1, 3),
                (2, 0), (2, 2), (2, 4),
                (3, 1), (3, 3), (4, 2)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def get_new_pos(old_board, new_state):
    new_board = new_state.get_board().get_board_state()
    diff = new_board == old_board
    x, y = np.where(diff == False)
    result = []
    for x_i, y_i in zip(x, y):
        result.append((x_i, y_i))
    return result

def to_str(state, action):
    res = str(action)
    board = state.get_board().get_board_state()
    for row in board:
        for slot in row:
            res += str(slot)
    return res

class AI(Player):
    in_hand = 12
    score = 0
    name = "smart_agent_half"
    state = None

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.remaining_time = -1
        self.turn_start_time = -1
        # self.opponent_remaining_time = -1
        # self.opponent_turn_start_time = -1
        self.TIME_SLOPE = 0.025
        self.TIME_INTERCEPT = 0.01
        self.DEPTH_THRESHOLD = 20
        self.TIME_THRESHOLD = 10  # in seconds
        self.prev_action = None  # (0, [])
        self.previous_board = None
        self.drops = []
        self.state = 0
        self.closed = {}

    def update_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            # print("=====", self.state, "=====")

    # def add_prev_action(self, action):
    #     if len(self.prev_actions[1]) < 1:
    #         self.prev_actions = (0, [action, None, None, None, None])
    #     else:
    #         self.prev_actions[1][(self.prev_actions[0] + 1) % 5] = action
    #         self.prev_actions = ((self.prev_actions[0] + 1) % 5, self.prev_actions[1])

    def isBoringMove(self, prev_score, new_action, new_score):
        if self.prev_action is None:
            return False
        return self.prev_action.action['at'] == new_action.action['to'] and \
               self.prev_action.action['to'] == new_action.action['at'] and prev_score == new_score

    def compute_time_threshold(self, state):
        # if self.remaining_time < 0.925 * self.opponent_remaining_time:
        #     self.TIME_SLOPE *= 0.5  # OLD = 0.666
        # else:
        #     self.TIME_SLOPE += 0.001
        # self.TIME_INTERCEPT = max(0.001 * self.remaining_time, 0.01)
        # self.TIME_SLOPE = max(self.TIME_SLOPE, 0.005)  # OLD = 0.01
        mult = 6.0
        intercept = (10.0 - mult) / 2.0
        timing = (mult * sigmoid(-50.0 + (self.remaining_time / 5.0))) + intercept

        if state.MAX_SCORE - state.score[self.color.value] < 3:
            return timing * 2
        else:
            return timing  # 0.025 * self.remaining_time  # self.TIME_SLOPE * self.remaining_time + self.TIME_INTERCEPT

    # def update_times(self, state, remain_time):
    #     if self.opponent_remaining_time == -1:
    #         self.opponent_remaining_time = remain_time
    #     else:
    #         elapsed = 0.002 * (
    #                 time.time() - self.opponent_turn_start_time) if state._latest_player != self.color.value else 0
    #         # print("Estimated elapsed time:", elapsed)
    #         self.opponent_remaining_time -= elapsed
    #     # print("Estimated opponent remaining time:", self.opponent_remaining_time)
    #     self.TIME_THRESHOLD = self.compute_time_threshold()

    def play(self, state, remain_time):
        # print("")
        # print(f"Player {self.position} is playing.")
        # print("time remain is ", remain_time, " seconds")

        # self.update_times(state, remain_time)
        self.remaining_time = remain_time
        self.TIME_THRESHOLD = self.compute_time_threshold(state)

        if state.phase == 1:
            if self.previous_board is None:
                self.previous_board = np.full_like(state.get_board().get_board_state(), Color.empty)
            new_action = self.drop_phase(state)
            new_state = deepcopy(state)
            SeegaRules.act(new_state, new_action, self.color.value)
            self.previous_board = new_state.get_board().get_board_state()
            # self.opponent_turn_start_time = time.time()
            return new_action
        else:
            succs = self.successors(state)
            if succs is not None and len(succs) < 2:
                return succs[0][0]

            self.turn_start_time = time.time()
            # minimax_result = minimax_search(state, self)
            iterative_deepening_result = self.iterative_deepening(state)
            # self.opponent_turn_start_time = time.time()
            if isinstance(iterative_deepening_result, SeegaAction):
                return iterative_deepening_result
            else:
                print("Prout", iterative_deepening_result)
                return SeegaRules.random_play(state, self.color.value)  # minimax_result

    def iterative_deepening(self, state):
        # now = time.time()
        best = None

        for self.DEPTH_THRESHOLD in range(100000):
            print(f"{self.name} ==== Iterative deepening with depth = {self.DEPTH_THRESHOLD} ==== ")
            minimax_result = minimax_search(state, self)
            if best is None or best[0] < minimax_result[0]:
                best = minimax_result
            self.DEPTH_THRESHOLD += 1
            now = time.time()
            print(f"{self.name} -> time since beginning of the turn: {now - self.turn_start_time}")
            if (now - self.turn_start_time) > self.TIME_THRESHOLD:
                break

        # self.DEPTH_THRESHOLD = 1
        # while (now - self.turn_start_time) < self.TIME_THRESHOLD and self.DEPTH_THRESHOLD < 20:
        #     minimax_result = minimax_search(state, self)
        #     if best is None or best[0] < minimax_result[0]:
        #         best = minimax_result
        #     self.DEPTH_THRESHOLD += 1
        #     now = time.time()

        self.prev_action = best[1]
        return best[1]

    def drop_phase(self, state):
        if self.color == Color.green:
            return self.best_drop_phase_1(state)
        elif self.color == Color.black:
            return self.best_drop_phase_2(state)
            # return self.symmetric_drop_phase(state)
        else:
            return SeegaRules.random_play(state, self.color.value)

    def symmetric_drop_phase(self, state):

        def flatten_pos(pos):
            new_pos = pos[0] * self.previous_board.shape[0] + pos[1]
            return new_pos

        def unflatten_pos(pos):
            tmp = pos % self.previous_board.shape[0]
            new_pos = int((pos - tmp) / self.previous_board.shape[0]), tmp
            return new_pos

        def get_symmetric(flattened_pos):
            return 24 - flattened_pos

        comparison = state.get_board().get_board_state() == self.previous_board
        if comparison.all():
            return SeegaRules.random_play(state, self.color.value)

        if len(self.drops) < 1:
            positions = get_new_pos(self.previous_board, state)
            for pos in positions:
                flattened_pos = flatten_pos(pos)
                sym_flat_pos = get_symmetric(flattened_pos)
                sym_pos = unflatten_pos(sym_flat_pos)
                self.drops.append(sym_pos)

        popped = self.drops.pop()

        actions = SeegaRules.get_player_actions(state, self.color.value)
        for action in actions:
            new_state = deepcopy(state)
            SeegaRules.act(new_state, action, self.color.value)
            positions = get_new_pos(state.get_board().get_board_state(), new_state)
            if positions[0] == popped:
                return action

        return SeegaRules.random_play(state, self.color.value)

    def best_drop_phase_1(self, state):

        def get_action_from_pos(pos):
            actions = SeegaRules.get_player_actions(state, self.color.value)
            for a in actions:
                new_state = deepcopy(state)
                SeegaRules.act(new_state, a, self.color.value)
                positions = get_new_pos(state.get_board().get_board_state(), new_state)
                if positions[0] == pos:
                    return a
            return None

        def state_0():
            print("state 0")
            random.shuffle(edge_middle)
            while len(edge_middle) > 0:
                pos = edge_middle.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_1():
            print("state 1")
            random.shuffle(around_center)
            while len(around_center) > 0:
                pos = around_center.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    around_center.clear()
                    return a
            return None

        def state_2():
            print("state 2")
            random.shuffle(corners)
            while len(corners) > 0:
                pos = corners.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_3():
            print("state 3")
            random.shuffle(next_to_corners)
            while len(next_to_corners) > 0:
                pos = next_to_corners.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        switcher = {
            0: lambda: state_0(),
            1: lambda: state_1(),
            2: lambda: state_2(),
            3: lambda: state_3()
        }

        action = None
        while action is None and self.state < 4:
            print("self.state:", self.state)
            action = switcher[self.state]()
            if action is None:
                self.state += 1

        if action is not None:
            return action
        else:
            print("state 4")
            return self.symmetric_drop_phase(state)

    def best_drop_phase_2(self, state):

        def get_action_from_pos(pos):
            actions = SeegaRules.get_player_actions(state, self.color.value)
            for a in actions:
                new_state = deepcopy(state)
                SeegaRules.act(new_state, a, self.color.value)
                positions = get_new_pos(state.get_board().get_board_state(), new_state)
                if positions[0] == pos:
                    return a
            return None

        def state_0():
            print("state 0")
            board = state.get_board().get_board_state()
            for p1, p2 in next_to_center:
                act = None
                if board[p1[0]][p1[1]] == Color.empty and board[p2[0]][p2[1]] == Color.empty:
                    act = get_action_from_pos(p1)
                elif board[p1[0]][p1[1]] == self.color and board[p2[0]][p2[1]] == Color.empty:
                    act = get_action_from_pos(p2)
                    next_to_center.clear()
                if act is not None:
                    return act
            return None

        def state_1():
            print("state 1")
            random.shuffle(corners)
            while len(corners) > 0:
                pos = corners.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_2():
            print("state 2")
            random.shuffle(next_to_corners)
            while len(next_to_corners) > 0:
                pos = next_to_corners.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_3():
            print("state 3")
            random.shuffle(to_not_avoid)

        switcher = {
            0: lambda: state_0(),
            1: lambda: state_1(),
            2: lambda: state_2(),
            3: lambda: state_3()
        }

        action = None
        while action is None and self.state < 4:
            print("self.state:", self.state)
            action = switcher[self.state]()
            if action is None:
                self.state += 1

        if action is not None:
            return action
        else:
            print("state 4")
            while len(to_not_avoid) > 0:
                pos = to_not_avoid.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return self.symmetric_drop_phase(state)

    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """

    def successors(self, state):
        succs = []
        actions = SeegaRules.get_player_actions(state, self.color.value)

        for action in actions:
            new_state = deepcopy(state)
            SeegaRules.act(new_state, action, self.color.value)
            pair = to_str(new_state, action)
            if pair not in self.closed:  # and not self.isBoringMove(state.score, action, new_state.score):
                self.closed[pair] = 1
                succs.append((action, new_state))

        if len(succs) < 1:
            return None
        # if len(succs) < 1 and len(actions) > 0:
        #     for action in actions:
        #         new_state = deepcopy(state)
        #         SeegaRules.act(new_state, action, self.color.value)
        #         succs.append((action, new_state))

        succs.sort(key=lambda x: self.evaluate(x[1]), reverse=True)
        # succs = succs[:math.ceil(len(succs) * 0.75)]
        # random.shuffle(succs)
        # returned = succs[:math.ceil(len(succs) * 0.8)]

        return succs  # returned

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """

    def cutoff(self, state, depth):
        now = time.time()
        # print("depth:", depth)
        depth_condition = depth > self.DEPTH_THRESHOLD
        time_condition = now - self.turn_start_time > self.TIME_THRESHOLD
        condition = SeegaRules.is_end_game(state) or time_condition or depth_condition
        return condition

    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """

    def evaluate(self, state):
        if state.score[-1 * self.color.value] >= state.MAX_SCORE - 1 > state.score[self.color.value]:
            return -1 * float("inf")

        if SeegaRules.is_end_game(state) and state.score[self.color.value] > state.score[-1 * self.color.value]:
            return float("inf")

        board = state.get_board().get_board_state()

        player_density = (state.MAX_SCORE - state.score[-1 * self.color.value]) / state.MAX_SCORE
        opponent_density = (state.MAX_SCORE - state.score[self.color.value]) / state.MAX_SCORE
        density = (player_density + opponent_density) / 2

        defensive_score = state.MAX_SCORE - state.score[-1 * self.color.value]
        aggressive_score = state.score[self.color.value]

        x_center, y_center = int((len(board) - 1) / 2), int((len(board[1]) - 1) / 2)

        if density > 2 / 3:
            if player_density < 0.6 * opponent_density:
                self.update_state("More Defensive")
                score = 4 * defensive_score + aggressive_score
            elif player_density < 0.8 * opponent_density:
                self.update_state("Defensive")
                score = 2 * defensive_score + aggressive_score
            else:
                self.update_state("Neutral")
                score = defensive_score + aggressive_score
        else:
            if opponent_density < 0.6 * player_density:
                self.update_state("More Aggressive")
                score = defensive_score + 4 * aggressive_score
            elif opponent_density >= player_density:
                self.update_state("Neutral")
                score = defensive_score + aggressive_score
            else:
                self.update_state("Aggressive")
                score = defensive_score + 2 * aggressive_score

        if density > 2 / 3 and player_density > opponent_density:
            for (x, y) in [(0, 0), (0, -1), (-1, 0), (-1, -1), (x_center, y_center)]:
                score = (score * 1.1 if (x, y) == (x_center, y_center) else score * 1.05) \
                    if board[x][y] == self.color else score

        return score

    """
    Specific methods for a Seega player (do not modify)
    """

    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos['in_hand']
        self.score = infos['score']

    def reset_player_informations(self):
        self.in_hand = 12
        self.score = 0


"""
MiniMax and AlphaBeta algorithms.
Adapted from:
    Author: Cyrille Dejemeppe <cyrille.dejemeppe@uclouvain.be>
    Copyright (C) 2014, Universite catholique de Louvain
    GNU General Public License <http://www.gnu.org/licenses/>
"""

inf = float("inf")


def minimax_search(state, player, prune=True):
    """Perform a MiniMax/AlphaBeta search and return the best action.
    Arguments:
    state -- initial state
    player -- a concrete instance of class AI implementing an Alpha-Beta player
    prune -- whether to use AlphaBeta pruning
    """

    def max_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        val = -inf
        action = None
        succs = player.successors(state)
        if succs is not None:
            for a, s in succs:
                if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                    v, _ = max_value(s, alpha, beta, depth + 1)
                else:  # next turn is for the other one
                    v, _ = min_value(s, alpha, beta, depth + 1)
                if v > val:
                    val = v
                    action = a
                    if prune:
                        if v >= beta:
                            return v, a
                        alpha = max(alpha, v)
            return val, action
        else:
            return player.evaluate(state), None

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        val = inf
        action = None
        succs = player.successors(state)
        if succs is not None:
            for a, s in succs:
                if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                    v, _ = min_value(s, alpha, beta, depth + 1)
                else:  # next turn is for the other one
                    v, _ = max_value(s, alpha, beta, depth + 1)
                if v < val:
                    val = v
                    action = a
                    if prune:
                        if v <= alpha:
                            return v, a
                        beta = min(beta, v)
            return val, action
        else:
            return player.evaluate(state), None

    score, action = max_value(state, -inf, inf, 0)
    # player.add_prev_action(action)
    return score, action
