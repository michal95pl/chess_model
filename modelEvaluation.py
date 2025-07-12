import chess.engine as chess_engine
import chess.pgn

from chessModel import ChessNet
from boardPlus import BoardPlus
from chess_mctsnn import AMCTS
import numpy as np
import torch

engine = chess_engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
engine.configure({"Skill Level": 2})

net = ChessNet(80, 30, device="cpu")
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

net.load_state_dict(torch.load("learn_files/chess_model3_epoch12.pt"))
optimizer.load_state_dict(torch.load("learn_files/chess_optimizer3_epoch12.pt"))

board = BoardPlus()

pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Stockfish vs MCTS"
pgn_game.headers["White"] = "Stockfish"
pgn_game.headers["Black"] = "MCTS"

i = 0
while True:
    result = engine.play(board, chess_engine.Limit(time=1.0))
    move = result.move
    board.push(move)
    pgn_game.add_variation(move)

    if board.is_game_over():
        print("Game Over")
        print(board.result())
        break

    prob = AMCTS(200, net).search(board)
    move_id = np.argmax(prob)
    move_mcts = board.decode_move(move_id)
    move_mcts = board.change_move_perspective(move_mcts)
    board.push(move_mcts)
    pgn_game.add_variation(move_mcts)
    i += 1
    if board.is_game_over():
        print("Game Over")
        print(board.result())
        break

print(f"Total moves: {i}")

print("PGN Game:")
print(pgn_game)

engine.quit()



