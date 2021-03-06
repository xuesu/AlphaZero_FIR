# -*- coding: UTF-8 -*-
import os

ROOT_DIR_PATH = os.path.join(*(os.path.split(os.path.abspath(__file__))[:-1]))

FIR_W = 6  # 五子棋游戏棋盘宽度，标准为13，但训练速度过慢
FIR_H = 6  # 五子棋游戏棋盘高度
FIR_BOARD_SZ = FIR_H * FIR_W  # 五子棋游戏棋盘格数
FIR_SUCCESS_NUM = 4  # 连满几颗获胜
C_U_MCTS = 5.0  # MCTS新结点探索偏重
MCTS_UPDATE_NUM = FIR_BOARD_SZ * 20  # 每次MCTS搜索执行多少次拓展
MAX_STEP_UPHEAVAL = 100  # 游戏最大步数
NET_FILTER_NUM = 32  # 网络
NET_LAYER_NUM = 1  # 残差网络层数
NET_BATCH_LEN = 256  # 训练集长度
NET_ACT_LAST_CNN_FILTER_NUM = 2  # 残差网络后概率网络最后一层卷积 filters
NET_V_LAST_CNN_FILTER_NUM = 4  # 残差网络后价值网络最后一层卷积 filters
NET_V_LAST_LINEAR_UNIT_NUM = FIR_BOARD_SZ  # 残差网络后价值网络第一层线性层 filters
TRAIN_STEP_PER_EVALUATE = 50  # 训练对局次数
TRAIN_STEP = 100000  # 训练对局次数
TRAIN_PER_EPOCH = 5
EPOCH_PER_TRAIN = 10
EVALUATE_STEP = 11  # 评估（是否保存模型）对局次数
LOG_PATH = os.path.join(ROOT_DIR_PATH, "logs")
MODEL_PATH = os.path.join(ROOT_DIR_PATH, "models", "model")
