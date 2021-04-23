from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
from time import time

"""
python main.py -ai0 ./seega/random_agent.py -ai1 agents/smart_agent.py -s 0.5 -t 120
"""

class AI(Player):

    in_hand = 12
    score = 0
    name = "SMART"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.clock = 0
        self.turn_start_time = 0

    def play(self, state, remain_time):
        # print("")
        # print(f"Player {self.position} is playing.")
        # print("time remain is ", remain_time, " seconds")

        # if state.phase == 1:
        #     player = self.position
        #     print("phase 1")
        #     return SeegaRules.random_play(state, player)
        # else:
        self.clock = remain_time
        self.turn_start_time = time()
        return minimax_search(state, self)

    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        successors = []
        player = self.position
        for action in SeegaRules.get_player_actions(state, player):
            potential = deepcopy(state)
            SeegaRules.act(potential, action, player)
            suc = (action,potential)
            successors.append(suc)
        # print("nb succ : "+ str(len(successors)) )
        # print(successors)

        return successors

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        max_depth = 3
        max_turn_time = round(self.clock * 0.01)
        time_elapsed = self.turn_start_time - time()

        time_finished = time_elapsed >= max_turn_time
        return time_finished or depth > max_depth or SeegaRules.is_end_game(state)

    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        if state.phase == 1:
            sum = 0
            cptMiddle = 0
            for x,y in state.board.get_player_pieces_on_board(self.color):
                if (x,y) == (0,2) or (x,y) == (2,0) or (x,y) == (4,2) or (x,y) == (2,4):
                    sum += 3
                elif cptMiddle < 2 and ((x,y) == (1,2) or (x,y) == (2,1) or (x,y) == (3,2) or (x,y) == (2,3)):
                    cptMiddle+=1
                    if state.board.get_cell_color(pair_moves((x,y))) == Color.green:
                        sum += 6
                    else:
                        sum += 3
                elif x == 0 or x == 4 or y == 0 or y == 4:
                    sum += 2
                else:
                    sum += 1
            return sum
        else:
            player = self.position
            opponent = self.position * (-1)
            boring = state.boring_moves
            return (state.score[player] + (state.MAX_SCORE - state.score[opponent])) - boring

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

def pair_moves(cell):
    x,y = cell
    if (x,y) == (1,2):
        return 4,2
    if (x,y) == (3,2):
        return 0,2
    if (x,y) == (2,1):
        return 2,4
    if (x,y) == (2,3):
        return 2,0

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
            else:                                             # next turn is for the other one
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
            else:                                             # next turn is for the other one
                v, _ = max_value(s, alpha, beta, depth + 1)
            if v < val:
                val = v
                action = a
                if prune:
                    if v <= alpha:
                        return v, a
                    beta = min(beta, v)
        return val, action

    _, action = max_value(state, -inf, inf, 0)
    return action