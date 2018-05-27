import abc


class Player(object):
    def __init__(self, ind, mgame):
        self.ind = ind
        self.mgame = mgame

    @abc.abstractclassmethod
    def nxt_move(self):
        pass

    @abc.abstractclassmethod
    def game_ended(self):
        pass

    @abc.abstractclassmethod
    def other_nxt_move(self, action):
        pass


class HumanPlayer(Player):
    def __init__(self, ind, mgame):
        super(HumanPlayer, self).__init__(ind, mgame)

    def show_board(self):
        print('_' + ''.join([chr(ch) for ch in range(ord('A'), ord('A') + self.mgame.w)]))
        for x in range(self.mgame.h):
            print(str(x + 1) + ''.join(
                ['_' if self.mgame.board[x][y] == -1 else "%.0f" % self.mgame.board[x][y] for y in
                 range(self.mgame.w)]))

    def get_nxt_action(self):
        action = -1
        while not self.mgame.is_valid_action(action):
            cmd = input(f"Your id is {self.ind}\nPlease input Next Move: (eg. 1 A)")
            x = ord(cmd[0]) - ord('1')
            y = ord(cmd[2].upper()) - ord('A')
            action = x * self.mgame.w + y
        print("OK")
        return action

    def nxt_move(self):
        self.show_board()
        return self.get_nxt_action()

    def other_nxt_move(self, action):
        print("The other player input {} {}".format(1 + action // self.mgame.w, chr(action % self.mgame.w + ord('A'))))

    def game_ended(self):
        if self.mgame.winner == -1:
            print("TIED!")
        elif self.mgame.winner == self.ind:
            print("YOU WIN!")
        else:
            print(f"YOU LOSE!\nThe winner is {self.mgame.winner}")
