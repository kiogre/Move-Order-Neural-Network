from MLChess import encode_board, encode_legal_moves, JellyFishPointer, ChessPointerExplainer
import chess
import torch
import matplotlib.pyplot as plt

# Carica il modello
model = JellyFishPointer()
ckpt  = torch.load("./checkpoints_az_v3/last.pt", map_location="cpu")
state_dict = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

explainer = ChessPointerExplainer(model)

# Prepara una posizione
fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
board = chess.Board(fen)

board_tensor = encode_board(fen)
legal_moves_tensor = encode_legal_moves(board)
move_mask = torch.ones(legal_moves_tensor.shape[0], dtype=torch.bool)

# Analisi completa
fig = explainer.complete_analysis(
    board_tensor, 
    legal_moves_tensor, 
    move_mask, 
    board=board,
    save_path='pointer_analysis.png'
)
plt.show()