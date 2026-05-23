import chess
import torch
from MLChess import encode_board, encode_legal_moves, JellyFishPointer

CHECKPOINT = "checkpoints_az_v2/last.pt"
DEVICE = torch.device("cuda")

model = JellyFishPointer().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

@torch.no_grad()
def get_value(fen):
    board_t = encode_board(fen).unsqueeze(0).to(DEVICE)
    moves_t = encode_legal_moves(chess.Board(fen)).unsqueeze(0).to(DEVICE)
    _, _, value = model(board_t, moves_t)
    return value[0,0].item()

# Posizione iniziale — dovrebbe essere vicino a 0
print(f"Posizione iniziale (bianco muove): {get_value(chess.STARTING_FEN):.4f}")

# Stessa posizione ma col nero che deve muovere — dovrebbe essere simmetrica
board = chess.Board()
board.turn = chess.BLACK
print(f"Posizione iniziale (nero muove):   {get_value(board.fen()):.4f}")

# Vantaggio materiale bianco — dovrebbe essere positivo quando muove bianco
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")  # bianco senza un cavallo
print(f"Bianco senza cavallo (bianco muove): {get_value(board.fen()):.4f}")

# Stesso svantaggio ma visto dal nero
board = chess.Board("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")  # nero senza un cavallo  
print(f"Nero senza cavallo (nero muove):     {get_value(board.fen()):.4f}")