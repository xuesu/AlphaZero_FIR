import game
import mes
import symbol


def evaluate(value_net):
    value_net.loggable = False
    print("Evaluating...")
    other_game = game.FIRGame()
    other_value_net = symbol.ValueNet(other_game, renew=False, loggable=False)
    win_num = 0
    tie_num = 0
    lose_num = 0
    score = 0
    for i in range(mes.EVALUATE_STEP):
        other_game.reset()
        other_game.add_player(value_net=value_net if i % 2 == 0 else other_value_net, is_human=False, trainable=False)
        other_game.add_player(value_net=other_value_net if i % 2 == 0 else value_net, is_human=False, trainable=False)
        other_game.play()
        if other_game.winner == -1:
            tie_num += 1
        elif other_game.winner == i % 2:
            win_num += 1
            score += 1
        else:
            lose_num += 1
            score -= 1
    print(f"Win:{win_num}, Tie:{tie_num}, Lose:{lose_num}")
    value_net.loggable = True
    return score


def train():
    mgame = game.FIRGame()
    value_net = symbol.ValueNet(mgame)
    for i in range(1, mes.TRAIN_STEP):
        mgame.reset()
        mgame.add_player(value_net=value_net, is_human=False, trainable=True)
        mgame.add_player(value_net=value_net, is_human=False, trainable=True)
        mgame.play()
        if i % mes.TRAIN_STEP_PER_EVALUATE == 0 and evaluate(value_net) > 0 :
            print("Saving...")
            value_net.save()


if __name__ == '__main__':
    train()
