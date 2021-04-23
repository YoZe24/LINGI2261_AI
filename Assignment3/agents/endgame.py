
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 15, 10:48:24
@last modified : 2021 Apr 22, 14:42:50
"""

import numpy as np
from core.player import Player, Color
from seega.seega_rules import SeegaRules
from seega.seega_actions import SeegaAction, SeegaActionType
from copy import deepcopy
from time import perf_counter
from collections import defaultdict


import os
import math
import torch
import torch.utils.model_zoo
import logging
import numpy as np
from enum import Enum

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


class ContestAgent:
    _color_to_int = np.vectorize(lambda color: color.value)

    def __init__(self, model_url: str):
        self._model = torch.utils.model_zoo.load_url(
            model_url, map_location="cpu"
        ).eval()

    def _action_from_1D_to_3D(self, index: int):
        z = index // (self.nrow * self.ncol)
        index -= z * self.nrow * self.ncol
        y = index // self.nrow
        x = index % self.nrow
        return x, y, z

    def _action_from_3D_to_SeegaAction(self, action):
        x, y, z = action
        at = (x, y)
        dx, dy = Action.get_dx(z)
        to = (x + dx, y + dy)
        return SeegaAction(SeegaActionType.MOVE, at=at, to=to)

    def _state_to_ndarray(self, state):
        return self._color_to_int(state.board.get_board_state())

    def __call__(self, state):
        self.nrow, self.ncol = state.board.board_shape
        board = self._state_to_ndarray(state)
        inputs = {"obs": torch.tensor(board).view(1, -1)}
        with torch.no_grad():
            outputs = self._model(inputs)[0][0]

        return outputs.numpy()

    def _state_to_actions(self, state, reverse):
        self.nrow, self.ncol = state.board.board_shape
        board = self._state_to_ndarray(state)
        inputs = {"obs": torch.tensor(board).view(1, -1)}
        with torch.no_grad():
            outputs = self._model(inputs)[0][0]
            actions_sorted = torch.argsort(outputs, descending=reverse).numpy()
        actions_3D_sorted = list(map(self._action_from_1D_to_3D, actions_sorted))
        actions_SeegaAction_sorted = list(
            map(self._action_from_3D_to_SeegaAction, actions_3D_sorted)
        )
        return actions_SeegaAction_sorted


class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @staticmethod
    def get_dx(action):
        return [(-1, 0), (0, 1), (1, 0), (0, -1)][action]


class AI(Player):

    in_hand = 12
    score = 0
    name = "Contest Agent"
    _running_depth = 0  # Track the maximum depth in the iterative deepening
    _max_depth = -666

    _model_url = "https://github.com/RomainGrx/LINGI2261-projects/raw/master/Assignment%203/models/latest.pt"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.color = color
        self.oponent = Color.green if Color.green == color else Color.black
        self.position = color.value
        self._remaining_time = float("inf")
        self._total_time = None
        self._tracking_list = defaultdict(lambda: 0)
        self._agent = ContestAgent(model_url=self._model_url)

    def play(self, state, remain_time):
        print("")
        print(f"Player {self.position} is playing.")
        print("time remain is ", remain_time, " seconds")

        if self._total_time is None:
            self._total_time = remain_time
        self._remaining_time = remain_time

        def time_policy(policy="exponential", min_frac=1 / 2000, max_frac=1 / 100):
            """time_policy.
            :param policy: the desired policy between ('linear', 'exponential')
            :param min_frac: the minimum fraction of total time allowed for computing time
            :param max_frac: the maximum fraction of total time allowed for computing time
            """
            min, max = self._total_time * min_frac, self._total_time * max_frac

            # Calculate the linear and exponential policy
            logging.info(f"alpha time : {self._alpha_time}")
            schedulers = dict(
                linear=min + self._alpha_time * (max - min),
                exponential=min
                            + (np.exp(self._alpha_time * math.log(2)) - 1) * (max - min),
            )

            return schedulers[policy]

        self._max_running_time = time_policy("exponential")

        # Begining the search
        self._start_minimax = perf_counter()
        best_action = self.iterative_deepening(state)

        tracked_state = deepcopy(state)
        SeegaRules.act(tracked_state, best_action, self.position)
        self._track_state(tracked_state)

        return best_action

    @property
    def _running_time(self):
        return perf_counter() - self._start_minimax

    @property
    def _alpha_time(self):
        return self._remaining_time / self._total_time

    def _alpha_winning(self, state):
        """_alpha_winning.
        less than 0.5 means loosing
        greater than 0.5 means winning
        :param state: the current state
        """
        return (
                0.5
                + 0.5
                * (state.score[self.position] - state.score[-self.position])
                / state.MAX_SCORE
        )

    def _action_eq(self, a, b):
        return a.action_type == b.action_type and a.action == b.action

    def successors(self, state):
        """successors.
        The successors function must return (or yield) a list of pairs (a, s) in which a is the action played to reach the state s.
        :param state: the state for which we want the successors
        """
        next_player = state.get_next_player()
        is_our_turn = next_player == self.position
        successors = list()

        if not is_our_turn and self._already_tracked(state):
            return list()

        actions = SeegaRules.get_player_actions(state, next_player)
        if state.phase == 2:
            nn_actions = self._agent._state_to_actions(state, reverse=not is_our_turn)
            actions = list(
                filter(
                    lambda x: any([self._action_eq(x, a) for a in actions]),
                    nn_actions,
                )
            )

        for action in actions:
            next_state = deepcopy(state)
            if SeegaRules.act(next_state, action, next_player):
                successors.append((action, next_state))

        # Get all not already tracked states if loosing and is our turn
        if is_our_turn and self._alpha_winning(state) < 0.5:
            not_tracked = list(
                filter(lambda elem: not self._already_tracked(elem[1]), successors)
            )
            if not_tracked:
                successors = not_tracked

        return successors

    def cutoff(self, state, depth):
        """cutoff.
        The cutoff function returns true if the alpha-beta/minimax search has to stop and false otherwise.
        :param state: the state for which we want to know if we have to apply the cutoff
        :param depth: the depth of the cutoff
        """

        def timing_cutoff():
            return self._running_time > self._max_running_time

        def depth_cutoff():
            return depth > self._max_depth

        is_cutoff = False

        # Check if the game is at the end
        is_cutoff |= SeegaRules.is_end_game(state)

        # Get the cutoff from the current depth
        is_cutoff |= depth_cutoff()

        # Get the current cutoff from the time running the minimax
        is_cutoff |= timing_cutoff()

        # Track the maximum depth
        self._running_depth = max(self._running_depth, depth)

        return is_cutoff

    def evaluate(self, state):
        """evaluate.
        The evaluate function must return an integer value representing the utility function of the board.
        :param state: the state for which we want the evaluation scalar
        """
        cell_groups = dict(
            center=(2, 2),
            star_center=[(2, 1), (2, 3), (1, 2), (3, 2)],
            square_center=[(1, 1), (1, 3), (3, 1), (3, 3)],
            star_ext=[(2, 0), (2, 4), (0, 2), (4, 2)],
            square_ext=[(0, 0), (0, 4), (4, 0), (4, 4)],
        )

        def player_wins(player):
            return state.score[player] == state.MAX_SCORE

        def border_gravity(player):
            def is_border(cell):
                x, y = cell
                m, n = state.board.board_shape
                return x == 0 or y == 0 or x == m - 1 or y == n - 1

            return sum(
                list(
                    map(
                        is_border,
                        state.board.get_player_pieces_on_board(
                            Color.green if Color.green.value == player else Color.black
                        ),
                    )
                )
            )

        def evaluate_cells(color):
            def is_player_cell(cell, color):
                return state.board.get_cell_color(cell) == color

            def direct_center_score():
                score = 0.0
                for base_cell in cell_groups["star_center"]:
                    for oponent_cell, ext_cell in zip(
                            cell_groups["star_center"], cell_groups["star_ext"]
                    ):
                        if (
                                is_player_cell(base_cell, color)
                                and is_player_cell(oponent_cell, -color)
                                and is_player_cell(ext_cell, color)
                        ):
                            score += 1
                return score

            score = 0.0

            if state.phase == 1:
                score += direct_center_score()
            else:
                score += is_player_cell(cell_groups["center"], color)

            return score

        score = 0.0

        if state.phase == 1:

            score += evaluate_cells(self.position)
            score -= evaluate_cells(-self.position)

        elif state.phase == 2:

            # Self score
            score += state.score[self.position]
            score -= state.score[-self.position]

            score += evaluate_cells(self.color)
            score -= evaluate_cells(self.oponent)

            # score += border_gravity(self.position) / state.MAX_SCORE
            # score -= border_gravity(-self.position) / state.MAX_SCORE

            # Winning state
            if SeegaRules.is_end_game(state):
                score += 100 if self._alpha_winning(state) > 0.5 else -100

        return score

    def iterative_deepening(self, state):
        best_results = None

        self._max_depth = 0
        while True:
            results = minimax_search(state, self)
            if best_results is None or best_results[0] < results[0]:
                best_results = results
            if self._running_time > self._max_running_time:
                break
            self._max_depth += 1

        return best_results[1]

    def _hashable_state(self, state):
        board = state.board.get_json_board()
        list_board = tuple(tuple(l) for l in board)
        return list_board

    def _already_tracked(self, state):
        hashable = self._hashable_state(state)
        return hashable in self._tracking_list

    def _track_state(self, state):
        hashable = self._hashable_state(state)
        self._tracking_list[hashable] += 1

    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos["in_hand"]
        self.score = infos["score"]

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
            if (
                    s.get_latest_player() == s.get_next_player()
            ):  # next turn is for the same player
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
            if (
                    s.get_latest_player() == s.get_next_player()
            ):  # next turn is for the same player
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