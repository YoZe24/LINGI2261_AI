"""
Created on 26 oct. 12:02 2020

@author: HaroldKS
"""

import json


class SeegaState(object):

    def __init__(self, board, next_player=-1, boring_limit=50, game_phase=1):
        """The State of the Seega Game. It contains information regarding the game such as:
            - board          : The current board
            - score          : The game score
            - in_hand        : The number of piece in hand for each player
            - latest_move    : The latest performed action
            - latest_player  : The latest player
            - next_player    : The next player
            - just_stop      : The limit of non rewarding moves
            - boring_moves   : The current number of non rewarding moves

        Args:
            board (Board): The board game
            next_player (int, optional): The next or first play at the start. Defaults to -1.
            boring_limit (int, optional): Limit of non rewarding moves. Defaults to 200.
        """

        self.board = board
        self._latest_player = None
        self._latest_move = None
        self.captured = None
        self._next_player = next_player
        self.score = {-1: 0, 1: 0}
        board_shape = self.board.board_shape[0]
        if board_shape == 5:
            self.MAX_SCORE = 12
        elif board_shape > 5 and (board_shape == 7 or board_shape == 9):
            self.MAX_SCORE = 14

        self.in_hand = {-1: self.MAX_SCORE, 1: self.MAX_SCORE}
        self.boring_moves = 0
        self.just_stop = boring_limit
        self.phase = game_phase

    def get_board(self):
        return self.board

    def set_board(self, new_board):
        self.board = new_board

    def get_latest_player(self):
        return self._latest_player

    def get_latest_move(self):
        return self._latest_move

    def get_next_player(self):
        return self._next_player

    def set_latest_move(self, action):
        self._latest_move = action

    def set_next_player(self, player):
        self._next_player = player

    def set_latest_player(self, player):
        self._latest_player = player

    def get_player_info(self, player):
        return {'in_hand': self.in_hand[player],
                'score': self.score[player]}

    def get_json_state(self):
        json_state = {'latest_player': self.get_latest_player(),
                      'latest_move': self.get_latest_move(),
                      'next_player': self.get_next_player(),
                      'score': self.score,
                      'in_hand': self.in_hand,
                      'phase': self.phase,
                      'max_score': self.MAX_SCORE,
                      'boring_moves': self.boring_moves,
                      'just_stop': self.just_stop,
                      'board': self.board.get_json_board(),
                      }
        return json.dumps(json_state, default=str)
