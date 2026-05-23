import chess
import chess.svg
import torch
from MLChess import encode_board, encode_legal_moves, JellyFishPointer

CHECKPOINT = "checkpoints_az_v2/last.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JellyFishPointer().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

@torch.no_grad()
def get_move(board):
    board_t = encode_board(board.fen()).unsqueeze(0).to(DEVICE)
    moves_t = encode_legal_moves(board).unsqueeze(0).to(DEVICE)
    _, probs, value = model(board_t, moves_t)
    move_idx = probs[0].argmax().item()
    move = list(board.legal_moves)[move_idx]
    return move, value[0,0].item()

board = chess.Board()
print("Partita self-play greedy\n")

for i in range(150):
    if board.is_game_over():
        break
    move, value = get_move(board)
    print(f"{'Bianco' if board.turn == chess.WHITE else 'Nero':6} | mossa: {move} | valore stimato: {value:+.3f}")
    board.push(move)

print(f"\nRisultato: {board.result()}")
print(f"\nPGN:\n{board}")
