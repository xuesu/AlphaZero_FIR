import math
import random

import numpy

import mes
import player
import symbol


class TreeNode(object):
    def __init__(self, last_action):
        self.vsum = 0
        self.p = 1
        self.n = 0
        self.last_action = last_action
        self.children = dict()

    def get_search_value(self, par_sqrt_n):
        if self.n == 0:
            return mes.C_U_MCTS * self.p * par_sqrt_n
        return self.vsum / self.n + mes.C_U_MCTS * self.p * par_sqrt_n / (1 + self.n)


class MCTSPlayer(player.Player):
    def __init__(self, ind, mgame, value_net, trainable=True):
        super(MCTSPlayer, self).__init__(ind, mgame)
        if value_net is None:
            self.value_net = symbol.ValueNet(mgame=mgame)
        self.value_net = value_net
        self.curr_node = TreeNode(-1)
        self.pies_store = []
        self.boards_store = []
        self.trainable = trainable

    def update(self, node, player_id=None):
        if player_id is None:
            player_id = self.ind
        if node.n == 0 or not node.children:
            if self.mgame.k0() != -1:
                # winner is this node, however, not current player
                v = 1.0
                node.vsum = 0
            else:
                prob, v = self.value_net.predict(self.get_board4train(node.last_action, player_id))
                node.children = {action: TreeNode(action) for action in self.mgame.get_valid_actions()}
                prob[[action for action in range(prob.shape[0]) if action not in node.children]] = 0
                if numpy.sum(prob) > 0:
                    prob /= numpy.sum(prob)
                for action in node.children:
                    node.children[action] = TreeNode(action)
                    node.children[action].p = prob[action]
        else:
            par_sqrt_n = math.sqrt(node.n)
            child_search_values = {action: node.children[action].get_search_value(par_sqrt_n) for action in
                                   node.children}
            max_child_search_value = max(child_search_values.values())
            good_children = [action for action in child_search_values if
                             child_search_values[action] == max_child_search_value]
            selected_action = random.choice(good_children)
            assert self.mgame.is_valid_action(selected_action)
            self.mgame.board[selected_action // self.mgame.w][selected_action % self.mgame.w] = player_id
            v = self.update(node.children[selected_action], 1 - player_id)
            self.mgame.board[selected_action // self.mgame.w][selected_action % self.mgame.w] = -1
        node.vsum += v
        node.n += 1
        return -v

    def get_board4train(self, selected_action, curr_ind):
        my_board = numpy.zeros(self.mgame.board.shape)
        other_board = numpy.zeros(self.mgame.board.shape)
        selected_board = numpy.zeros(self.mgame.board.shape)
        my_board[self.mgame.board == curr_ind] = 1
        other_board[self.mgame.board == 1 - curr_ind] = 1
        if selected_action >= 0:
            selected_board[selected_action // self.mgame.w][selected_action % self.mgame.w] = 1
        return numpy.dstack([my_board, other_board, selected_board])

    def nxt_move(self):
        for _ in range(mes.MCTS_UPDATE_NUM):
            self.update(self.curr_node)
        prob = numpy.zeros([mes.FIR_BOARD_SZ])
        for action in self.curr_node.children:
            prob[action] = math.pow(self.curr_node.children[action].n, mes.TAO_N_MCTS)
        prob_sum = numpy.sum(prob)
        prob /= prob_sum
        if not self.trainable:
            max_prob = numpy.max(prob)
            selected_action = random.choice([i for i in range(self.mgame.board_sz) if prob[i] == max_prob])
        else:
            prob_valid = numpy.array(
                [1.0 if self.mgame.is_valid_action(action) else 0.0 for action in range(self.mgame.board_sz)])
            prob_valid /= numpy.sum(prob_valid)
            selected_action = numpy.random.choice([i for i in range(self.mgame.board_sz)],
                                                  p=prob_valid * 0.25 + prob * 0.75)
        self.curr_node = self.curr_node.children[selected_action]
        if self.trainable:
            self.mgame.board[selected_action // self.mgame.w][selected_action % self.mgame.w] = self.ind
            self.boards_store.append(self.get_board4train(selected_action, self.ind))
            self.mgame.board[selected_action // self.mgame.w][selected_action % self.mgame.w] = -1
            self.pies_store.append(prob)
        return selected_action

    def game_ended(self):
        if self.trainable:
            if self.mgame.winner == -1:
                return
            elif self.mgame.winner == self.ind:
                zs = [[1] for _ in range(len(self.boards_store))]
            else:
                zs = [[-1] for _ in range(len(self.boards_store))]
            self.value_net.train_step(self.boards_store, zs, self.pies_store)

    def other_nxt_move(self, action):
        self.curr_node = self.curr_node.children[action] \
            if self.curr_node.n != 0 and action in self.curr_node.children \
            else TreeNode(action)
