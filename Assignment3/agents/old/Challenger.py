import math

from core.player import Player, Color
from seega import SeegaState
from seega import SeegaAction
from seega.seega_rules import SeegaRules
from copy import deepcopy
from time import time

"""
python main.py -ai0 ./seega/random_agent.py -ai1 agents/smart_agent.py -s 0.5 -t 120
"""

class AI(Player):

    in_hand = 12
    score = 0
    name = "STONKS"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.clock = 0
        self.turn_start_time = 0
        self.phase1_time = 0.01
        self.phase2_time = 0.05
        self.depth = 20

    def play(self, state, remain_time):
        self.clock = remain_time
        self.turn_start_time = time()

        iterative_deepening_result = self.iterative_deepening(state)
        res: SeegaAction
        if isinstance(iterative_deepening_result, SeegaAction):
            res = iterative_deepening_result
        else:
            res = SeegaRules.random_play(state, self.color.value)
        return res
        #return minimax_search(state, self)

    def iterative_deepening(self, state):
        if state.phase == 1:
            max_turn_time = self.clock * self.phase1_time
            max_depth = 20
        else :
            max_turn_time = self.clock * self.phase2_time
            max_depth = 10

        best_move = None

        for self.depth in range(max_depth):
            minimax_result = minimax_search(state, self)
            if best_move is None or best_move[0] < minimax_result[0]:
                best_move = minimax_result
            self.depth += 1
            print(self.depth)
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
        opponent = player * (-1)
        for action in SeegaRules.get_player_actions(state, player):
            potential = deepcopy(state)
            SeegaRules.act(potential, action, player)

            if not SeegaRules.is_player_stuck(potential,opponent):
                suc = (action,potential)
                successors.append(suc)

        successors = list(set(successors))
        successors.sort(key=lambda x: self.evaluate(x[1]), reverse=True)
        successors = successors[:math.ceil(len(successors) * 0.25)]

        return successors

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        if state.phase == 1:
            max_turn_time = self.clock * self.phase1_time
            max_depth = 20
        else:
            max_turn_time = self.clock * self.phase2_time
            max_depth = 10

        time_elapsed = abs(self.turn_start_time - time())
        time_finished = time_elapsed >= max_turn_time
        return time_finished or depth > max_depth or SeegaRules.is_end_game(state)

    """def evaluate(self, state):
        player = self.position
        opponent = self.position * (-1)
        return (state.score[player] + (state.MAX_SCORE - state.score[opponent]))"""
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
                    for pair_move in pair_moves( (x,y) ):
                        if state.board.get_cell_color(pair_move) == Color.green:
                            if ++cell_cpt == 1: #first elem of pair_moves is the best move from 3 possibles opposite
                                board_value += 2
                            else:
                                board_value += 1

                elif x == 0 or x == 4 or y == 0 or y == 4:
                    board_value += 2
                else:
                    board_value += 1
            return board_value
        else:
            player = self.position
            opponent = self.position * (-1)
            boring = state.boring_moves
            op_color = Color(opponent)
            possibles_moves = ( (-1,0), (1,0), (0,-1), (0,1) )

            # print(state.get_json_state())
            killers = list()
            killeds = list()
            for cell in state.board.get_player_pieces_on_board(self.color):
                if state.board.is_center(cell):
                    board_value += 2
                    continue

                for neighbour_dif in possibles_moves:
                    neighbour = sum_cell(cell,neighbour_dif)

                    if state.board.get_cell_color(neighbour) == op_color:
                        board_value += 1
                        opposite_cell_bad = get_opposite(neighbour,cell)
                        if state.board.get_cell_color(opposite_cell_bad) == Color.empty: # if board !=  B G B  (this case is safe for the moment)
                            for potential_killer in possibles_moves: # if B G 0 and one B around 0 (0 = blanck) => opponent take next turn
                                killer = sum_cell(opposite_cell_bad,potential_killer)
                                if state.board.get_cell_color( killer ) == op_color:
                                    # print(str(neighbour) + " - " + str(cell) + " - " + str(opposite_cell_bad) + " bad killer : " +str(killer))

                                    if neighbour not in killeds:
                                        board_value -= 3
                                        killeds.append(cell)
                                        killers.append(killer)
                                    # else:
                                    # print("can't kill, cell or neighbour already tooked")
                                    break

                        if state.board.is_center(neighbour):
                            continue

                        opposite_cell_good = get_opposite(cell,neighbour)
                        if state.board.get_cell_color(opposite_cell_good) == Color.empty:
                            for potential_killer in possibles_moves:
                                killer = sum_cell(opposite_cell_good,potential_killer)
                                if state.board.get_cell_color(killer) == self.color:
                                    # print(str(neighbour) + " - " + str(cell) + " - " + str(opposite_cell_good) + " good killer : " +str(killer))
                                    killers.append(killer)
                                    killeds.append(neighbour)
                                    board_value += 3

                                    if cell in killeds:
                                        # print(str(cell) + "couldn't be killed, remove penalisation")
                                        board_value += 3
                                    if neighbour in killers:
                                        # print(str(neighbour) + " couldn't be killer, remove penalisation")
                                        board_value += 3
                                    break

                    if state.board.is_center(cell):
                        board_value += 2


            score = ((state.score[player] + (state.MAX_SCORE - state.score[opponent])) - boring) * 10 + board_value

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

def get_nearest_cell(cell,opponent_cells):
    min_dist = 999999
    for op_cell in opponent_cells:
        # op_cell = op_action.get_action_as_dict()["action"]["to"]
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
        for a, s in player.successors(state):
            if s.get_latest_player() == s.get_next_player():
                v, _ = max_value(s, alpha, beta, depth + 1)
            else:
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
            if s.get_latest_player() == s.get_next_player():
                v, _ = min_value(s, alpha, beta, depth + 1)
            else:
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
    # player.add_prev_action(action)
    return score, action