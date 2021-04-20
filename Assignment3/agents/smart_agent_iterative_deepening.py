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

def get_new_pos(old_board, new_state):
    new_board = new_state.get_board().get_board_state()
    diff = new_board == old_board
    x, y = np.where(diff == False)
    result = []
    for x_i, y_i in zip(x, y):
        result.append((x_i, y_i))
    return result


class AI(Player):
    in_hand = 12
    score = 0
    name = "iterative_deepening"
    state = None

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.remaining_time = -1
        self.START_TIME = -1
        self.TIME_SLOPE = 0.025
        self.TIME_INTERCEPT = 0.01
        self.DEPTH_THRESHOLD = 20
        self.TIME_THRESHOLD = 10  # in seconds
        self.previous_board = None
        self.drops = []
        self.state = 0

    def update_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            # print("=====", self.state, "=====")

    def compute_time_threshold(self):
        tmp = self.remaining_time / 90.0
        return 0.1 + 0.1 * np.exp(tmp) + 1.0

    def play(self, state, remain_time):
        # print("")
        # print(f"Player {self.position} is playing.")
        # print("time remain is ", remain_time, " seconds")

        self.TIME_THRESHOLD = self.compute_time_threshold()
        # print("Playing for", self.TIME_THRESHOLD, "s.")

        if state.phase == 1:
            if self.previous_board is None:
                self.previous_board = np.full_like(state.get_board().get_board_state(), Color.empty)
            new_action = self.drop_phase(state)
            new_state = deepcopy(state)
            SeegaRules.act(new_state, new_action, self.color.value)
            self.previous_board = new_state.get_board().get_board_state()
            return new_action
        else:
            self.remaining_time = remain_time
            self.START_TIME = time.time()
            now = self.START_TIME
            self.DEPTH_THRESHOLD = 1
            best = None
            while (now - self.START_TIME) < self.TIME_THRESHOLD and self.DEPTH_THRESHOLD < 15:
                minimax_result = minimax_search(state, self)
                if best is None or best[0] < minimax_result[0]:
                    # print("new best found")
                    best = minimax_result
                self.DEPTH_THRESHOLD += 1
                now = time.time()
            return best[1]

    def drop_phase(self, state):
        if self.color == Color.green:
            return self.best_drop_phase(state)
        elif self.color == Color.black:
            return self.symmetric_drop_phase(state)
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

    def best_drop_phase(self, state):

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
            while len(edge_middle) > 0:
                pos = edge_middle.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_1():
            print("state 1")
            while len(around_center) > 0:
                pos = around_center.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    around_center.clear()
                    return a
            return None

        def state_2():
            print("state 2")
            while len(corners) > 0:
                pos = corners.pop()
                a = get_action_from_pos(pos)
                if a is not None:
                    return a
            return None

        def state_3():
            print("state 3")
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
            result = SeegaRules.act(new_state, action, self.color.value)
            if type(result) is bool and not result:
                continue
            else:
                succs.append((action, new_state))
        random.shuffle(succs)
        return succs  # returned

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """

    def cutoff(self, state, depth):
        now = time.time()
        timeout_condition = now - self.START_TIME > self.TIME_THRESHOLD
        depth_condition = depth > self.DEPTH_THRESHOLD
        condition = SeegaRules.is_end_game(state) or timeout_condition or depth_condition
        return condition

    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """

    def evaluate(self, state):
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
            elif opponent_density < 0.8 * player_density:
                self.update_state("Aggressive")
                score = defensive_score + 2 * aggressive_score
            else:
                self.update_state("Neutral")
                score = defensive_score + aggressive_score

        if density > 2 / 3 and player_density > opponent_density:
            for (x, y) in [(0, 0), (0, -1), (-1, 0), (-1, -1), (x_center, y_center)]:
                score = (score * 1.1 if (x, y) == (x_center, y_center) else score * 1.05) if board[x][
                                                                                                 y] == self.color else score

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
        for a, s in player.successors(state):
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

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        val = inf
        action = None
        for a, s in player.successors(state):
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

    val, action = max_value(state, -inf, inf, 0)
    return val, action

