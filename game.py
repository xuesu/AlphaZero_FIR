import abc

import numpy

import mcts
import mes
import player


class Game(object):
    def __init__(self):
        self.max_step = -1
        self.w = -1
        self.h = -1
        self.board_sz = self.w * self.h
        self.board = None
        self.players = []
        self.winner = -1

    def valid_game_def(self):
        assert self.w < 26
        assert self.h < 26

    def in_board(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def is_valid_action(self, action):
        x = action // self.w
        y = action % self.w
        return self.in_board(x, y) and self.board[x][y] == -1

    def get_valid_actions(self):
        return [action for action in range(self.board_sz) if self.is_valid_action(action)]

    @abc.abstractclassmethod
    def play(self):
        pass

    @abc.abstractclassmethod
    def k0(self):
        pass

    @abc.abstractclassmethod
    def add_player(self, **kwargs):
        pass

    def reset(self):
        self.board = numpy.zeros(shape=[self.h, self.w]) - 1
        self.players = []
        self.board_sz = self.w * self.h
        self.winner = -1


class FIRGame(Game):
    def __init__(self):
        super(FIRGame, self).__init__()
        self.w = mes.FIR_W
        self.h = mes.FIR_H
        self.num = mes.FIR_SUCCESS_NUM
        self.max_step = min(self.w * self.h, mes.MAX_STEP_UPHEAVAL)
        self.tao = 1
        self.reset()

    def add_player(self, is_human=True, trainable=False, value_net=None):
        if is_human:
            self.players.append(player.HumanPlayer(len(self.players), self))
        else:
            self.players.append(mcts.MCTSPlayer(len(self.players), self, value_net, trainable))

    def k0(self):
        for x in range(self.h):
            for y in range(self.w):
                if self.board[x][y] == -1:
                    continue
                for dx, dy in [(1, 0), (0, 1), (1, 1)]:
                    fl = sum([1 for i in range(self.num) if
                              self.in_board(x + dx * i, y + dy * i) and self.board[x + dx * i][y + dy * i] ==
                              self.board[x][y]]) == self.num
                    if fl:
                        return self.board[x][y]
        return -1

    def play(self):
        """
            need to call reset before a new game start
        """
        for step_i in range(self.max_step):
            if step_i < 10:
                self.tao = 1
            else:
                self.tao = 10.0 / step_i
            player_id = step_i & 1
            player = self.players[player_id]
            action = player.nxt_move()
            if isinstance(player, mcts.MCTSPlayer) and player.value_net.loggable:
                print(f'Player{player_id}: Action: {action}')
            if not self.is_valid_action(action):
                # because now just consider 2 players
                print(f"Player: {player_id}, Action: {action} Did Not choose a valid action!")
                self.board[action // self.w][action % self.w] = player_id
                self.winner = 1 - player_id
            else:
                self.board[action // self.w][action % self.w] = player_id
                self.winner = self.k0()
            self.players[1 - player_id].other_nxt_move(action)
            if self.winner != -1:
                break
        print(f'Winner: {self.winner}')
        for player_id in range(len(self.players)):
            self.players[player_id].game_ended()
