import math
import random

import numpy

import mes
import player


class TreeNode(object):
    def __init__(self, last_action, parent_keys):
        self.vsum = 0
        self.p = 1
        self.n = 0
        self.last_action = last_action
        self.children = {action: None for action in parent_keys if action != last_action}

    def get_search_value(self, par_sqrt_n):
        if self.n == 0:
            return mes.C_U_MCTS * self.p * par_sqrt_n
        return self.vsum / self.n + mes.C_U_MCTS * self.p * par_sqrt_n / (1 + self.n)


class MCTSPlayer(player.Player):
    def __init__(self, ind, mgame, value_net, trainable=True):
        super(MCTSPlayer, self).__init__(ind, mgame)
        self.value_net = value_net
        self.curr_node = TreeNode(-1, self.mgame.get_valid_actions())
        self.pies_store = []
        self.boards_store = []
        self.trainable = trainable

    def update(self, node, mgame):
        if node.n == 0:
            prob, v = self.value_net.predict(self.get_board4train(mgame, node.last_action))
            prob[[action for action in range(prob.shape[0]) if action not in node.children]] = 0
            if numpy.sum(prob) > 0:
                prob /= numpy.sum(prob)
            for action in node.children:
                node.children[action] = TreeNode(action, list(node.children.keys()))
                node.children[action].p = prob[action]
                node.children[action].vsum = v
        else:
            search_value_max = 0
            par_sqrt_n = math.sqrt(node.n)
            selected_action = list(node.children.keys())[0]
            for action in node.children:
                child_search_value = node.children[action].get_search_value(par_sqrt_n)
                if child_search_value > search_value_max \
                        or (child_search_value == search_value_max and random.random() <= 1.0 / len(node.children)):
                    selected_action = action
                    search_value_max = node.children[action].get_search_value(par_sqrt_n)
            v = self.update(node.children[selected_action], mgame)
        node.vsum += v
        node.n += 1
        return v

    def get_board4train(self, mgame, selected_action):
        my_board = numpy.zeros(mgame.board.shape)
        other_board = numpy.zeros(mgame.board.shape)
        selected_board = numpy.zeros(mgame.board.shape)
        my_board[mgame.board == self.ind] = 1
        other_board[mgame.board == 1 - self.ind] = 1
        if selected_action >= 0:
            selected_board[selected_action // mgame.w][selected_action % mgame.w] = 1
        return numpy.dstack([my_board, other_board, selected_board])

    def nxt_move(self):
        for _ in range(mes.MCTS_UPDATE_NUM):
            self.update(self.curr_node, self.mgame)
        prob = numpy.zeros([mes.FIR_BOARD_SZ])
        for action in self.curr_node.children:
            prob[action] = math.pow(self.curr_node.children[action].n, mes.TAO_N_MCTS)
        prob_sum = numpy.sum(prob)
        prob /= prob_sum
        if not self.trainable:
            selected_action = numpy.argmax(prob)
        else:
            random_v = random.random()
            selected_action = 0
            for action in range(prob.shape[0]):
                if random_v <= prob[action] and prob[action] > 0:
                    selected_action = action
                    break
                else:
                    random_v -= prob[action]
        self.curr_node = self.curr_node.children[selected_action]
        if self.trainable:
            self.boards_store.append(self.get_board4train(self.mgame, selected_action))
            self.pies_store.append(prob)
        return selected_action

    def game_ended(self):
        if self.trainable:
            if self.mgame.winner == -1:
                zs = [[0] for _ in range(len(self.boards_store))]
            elif self.mgame.winner == self.ind:
                zs = [[1] for _ in range(len(self.boards_store))]
            else:
                zs = [[-1] for _ in range(len(self.boards_store))]
            self.value_net.train_step(self.boards_store, zs, self.pies_store)

    def other_nxt_move(self, action):
        self.curr_node = self.curr_node.children[action] \
            if self.curr_node.n != 0 and action in self.curr_node.children \
            else TreeNode(action, self.mgame.get_valid_actions())
