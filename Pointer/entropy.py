import chess
import torch
from MLChess import encode_board, encode_legal_moves, JellyFishPointer

CHECKPOINT = "checkpoints_az/last.pt"
DEVICE = torch.device("cuda")

model = JellyFishPointer().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

board = chess.Board()  # posizione iniziale

with torch.no_grad():
    board_t = encode_board(board.fen()).unsqueeze(0).to(DEVICE)
    moves_t = encode_legal_moves(board).unsqueeze(0).to(DEVICE)
    _, probs, _ = model(board_t, moves_t)
    probs = probs[0].cpu()

legal_moves = list(board.legal_moves)
pairs = sorted(zip(probs.tolist(), legal_moves), reverse=True)

print(f"Max prob: {pairs[0][0]:.4f}  ({pairs[0][1]})")
print(f"Top 5:")
for p, m in pairs[:5]:
    print(f"  {m}: {p:.4f}")
print(f"Entropia: {-(probs * torch.log(probs + 1e-8)).sum().item():.4f}")