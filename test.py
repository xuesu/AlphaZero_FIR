import game
import symbol


def test():
    mgame = game.FIRGame()
    value_net = symbol.ValueNet(mgame, renew=False)
    for _ in range(2):
        op = input("AI?(Y/N)").strip().upper()
        if op == 'Y':
            mgame.add_player(value_net=value_net, is_human=False, trainable=False)
        else:
            mgame.add_player(is_human=True)
    mgame.play()


if __name__ == '__main__':
    test()
