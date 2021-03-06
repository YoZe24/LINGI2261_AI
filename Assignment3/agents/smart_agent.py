import math
import random

from collections import defaultdict

from core.player import Player, Color
from seega import SeegaState, SeegaAction
from seega.seega_rules import SeegaRules
from copy import deepcopy
from time import time

"""
python main.py -ai0 ./agents/smart_agent.py -ai1 agents/smart_agent.py -s 0.5 -t 120
"""

class AI(Player):

    in_hand = 12
    score = 0
    name = "Group 29"

    def __init__(self, color):
        super(AI, self).__init__(color)

        self.position = color.value
        self.clock = 0
        self.turn_start_time = 0
        self.phase1_time = 0.01
        self.phase2_time = 0.025
        self.depth = 0
        self._states = defaultdict(lambda: 0)


    def play(self, state, remain_time):

        self.clock = remain_time
        self.turn_start_time = time()
        iterative_deepening_result = self.iterative_deepening(state)
        res: SeegaAction
        self._states.clear()
        if isinstance(iterative_deepening_result, SeegaAction):
            res = iterative_deepening_result
        else:
            res = SeegaRules.random_play(state, self.color.value)
        return res


    def iterative_deepening(self, state):
        if state.phase == 1:
            max_turn_time = self.clock * self.phase1_time
            max_depth = 12
        else :
            max_turn_time = self.clock * self.phase2_time
            max_depth = 10

        best_move = None

        for self.depth in range(max_depth):
            minimax_result = minimax_search(state, self)
            if best_move is None or best_move[0] < minimax_result[0]:
                best_move = minimax_result
            self.depth += 1
            time_elapsed = abs(self.turn_start_time - time())
            time_finished = time_elapsed >= max_turn_time
            if time_finished:
                break

        return best_move[1]

    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        successors = []
        player = self.position
        opp = player * -1
        next_player = state.get_next_player()

        for action in SeegaRules.get_player_actions(state, next_player):
            potential = deepcopy(state)

            SeegaRules.act(potential, action, next_player)

            if state.phase == 2 and SeegaRules.is_player_stuck(state,opp) and SeegaRules.is_player_stuck(potential,opp):
                continue

            if not self._is_in_states(potential):
                suc = (action,potential)
                successors.append(suc)
            self._add_state(potential)

        if state.phase == 1:
            random.shuffle(successors)

        successors.sort(key=lambda x: self.evaluate(x[1]), reverse=next_player == player)
        # print(str(len(successors)) + " player turn : " + str(next_player))
        #
        # for suc in successors:
        #     print(str(suc[0].get_json_action()) + " = > "+suc[1].get_json_state())

        if len(successors) > 6:
            successors = successors[:math.ceil(len(successors) * 0.50)]

        return successors

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        if state.phase == 1:
            max_turn_time = self.clock * self.phase1_time
            max_depth = 12
        else:
            max_turn_time = self.clock * self.phase2_time
            max_depth = 10

        time_elapsed = abs(self.turn_start_time - time())
        time_finished = time_elapsed >= max_turn_time
        return time_finished or depth > max_depth or SeegaRules.is_end_game(state)

    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        board_value = 0
        if state.phase == 1:
            cpt_middle = 0
            for cell in state.board.get_player_pieces_on_board(self.color):
                x,y = cell
                if (x,y) == (0,2) or (x,y) == (2,0) or (x,y) == (4,2) or (x,y) == (2,4):
                    board_value += 3
                elif cpt_middle < 2 and ((x,y) == (1,2) or (x,y) == (2,1) or (x,y) == (3,2) or (x,y) == (2,3)):
                    cpt_middle+=1
                    board_value += 3
                    cell_cpt = 0
                    if self.position == Color.green:
                        for pair_move in pair_moves( (x,y) ):
                            if state.board.get_cell_color(pair_move) == Color.green:
                                if ++cell_cpt == 1: #first elem of pair_moves is the best move from 3 possibles opposite
                                    board_value += 2
                                else:
                                    board_value += 1
                    elif state.board.get_cell_color(back_pawn(cell)) == Color.black:
                        board_value += 2

                elif x == 0 or x == 4 or y == 0 or y == 4:
                    board_value += 2
                else:
                    board_value += 1
            return board_value
        else:
            player = self.position
            opponent = self.position * (-1)
            op_color = Color(opponent)
            possibles_moves = ( (-1,0), (1,0), (0,-1), (0,1) )

            killers = list()
            killeds = list()
            for cell in state.board.get_player_pieces_on_board(self.color):
                if state.board.is_center(cell):
                    board_value += 2

                for neighbour_dif in possibles_moves:
                    neighbour = sum_cell(cell,neighbour_dif)

                    if state.board.get_cell_color(neighbour) == op_color:
                        board_value += 1
                        opposite_cell_bad = get_opposite(neighbour,cell)

                        if state.board.get_cell_color(opposite_cell_bad) == Color.empty: # if board !=  B G B  (this case is safe for the moment)
                            if state.board.is_center(cell):
                                continue

                            for potential_killer in possibles_moves: # if B G 0 and one B around 0 (0 = blanck) => opponent take next turn
                                killer = sum_cell(opposite_cell_bad,potential_killer)
                                if state.board.get_cell_color( killer ) == op_color:

                                    if state.get_next_player() == player:
                                        value = 3
                                    else:
                                        value = 4

                                    if neighbour not in killeds:
                                        board_value -= value
                                        killeds.append(cell)
                                        killers.append(killer)
                                    break



                        opposite_cell_good = get_opposite(cell,neighbour)
                        if state.board.get_cell_color(opposite_cell_good) == Color.empty:
                            if state.board.is_center(neighbour):
                                continue
                            for potential_killer in possibles_moves:
                                killer = sum_cell(opposite_cell_good,potential_killer)
                                if state.board.get_cell_color(killer) == self.color:
                                    killers.append(killer)
                                    killeds.append(neighbour)

                                    if state.get_next_player() == player:
                                        value = 4
                                    else:
                                        value = 3

                                    board_value += value

                                    if state.get_next_player() == player:
                                        if cell in killeds:
                                            board_value += 3
                                        if neighbour in killers:
                                            board_value += 3
                                    break

            score = (state.score[player] + (state.MAX_SCORE - state.score[opponent])) * 10 + board_value

            if SeegaRules.is_end_game(state):
                score += 100 if state.score[player] > state.score[opponent] else -100

        return score

    def _hash_state(self, state):
        board = state.board.get_json_board()
        list_board = tuple(tuple(l) for l in board)
        return list_board

    def _is_in_states(self, state):
        return self._hash_state(state) in self._states

    def _add_state(self, state):
        self._states[self._hash_state(state)] += 1
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

def get_nearest_cell(cell,opponent_cells):
    min_dist = 999999
    for op_cell in opponent_cells:
        dist = abs(cell[0] - op_cell[0]) + abs(cell[1] - op_cell[1])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def get_opposite(cell_to_oppose, cell):
    if cell_to_oppose[0] == cell[0]:
        if cell_to_oppose[1] > cell[1]:
            return cell[0],cell[1]-1
        return cell[0],cell[1]+1

    if cell_to_oppose[1] == cell[1]:
        if cell_to_oppose[0] > cell[0]:
            return cell[0]-1,cell[1]
        return cell[0]+1, cell[1]

def sum_cell(cell1,cell2):
    return cell1[0] + cell2[0] , cell1[1] + cell2[1]

def back_pawn(cell):
    x,y = cell
    cellB = 0,2
    cellL = 2,0
    cellU = 4,2
    cellR = 2,4
    if (x,y) == (1,2):
        return cellB
    if (x,y) == (3,2):
        return cellU
    if (x,y) == (2,1):
        return cellL
    if (x,y) == (2,3):
        return cellR

def pair_moves(cell):
    x,y = cell
    cellB = 0,2
    cellL = 2,0
    cellU = 4,2
    cellR = 2,4
    if (x,y) == (1,2):
        return [cellU,cellL,cellR]
    if (x,y) == (3,2):
        return [cellB,cellL,cellR]
    if (x,y) == (2,1):
        return [cellR,cellU,cellB]
    if (x,y) == (2,3):
        return [cellL,cellU,cellB]

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
        # print("call max succ " + str(state.get_json_state()))
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
        # print("call min succ " + str(state.get_json_state()))
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

    score, action = max_value(state, -inf, inf, 0)
    return score,action