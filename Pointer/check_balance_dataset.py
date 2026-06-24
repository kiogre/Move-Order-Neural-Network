import pandas as pd
import chess

csv_file = "advantages_training_dataset.csv"  # aggiorna il percorso

df = pd.read_csv(csv_file)

mask = (
    df['Themes'].str.contains('advantage|crushing|endgame', na=False) &
    ~df['Themes'].str.contains('mateIn1', na=False)
)
df = df[mask].dropna(subset=["FEN", "Moves"])

white_to_move = 0
black_to_move = 0
errors = 0

for _, row in df.iterrows():
    try:
        board = chess.Board(row["FEN"])
        first_move = chess.Move.from_uci(row["Moves"].split()[0])
        board.push(first_move)
        # dopo la mossa dell'avversario, chi muove è il giocatore del puzzle
        if board.turn == chess.WHITE:
            white_to_move += 1
        else:
            black_to_move += 1
    except Exception:
        errors += 1

total = white_to_move + black_to_move
print(f"Posizioni totali: {total}")
print(f"Bianco muove: {white_to_move} ({100*white_to_move/total:.1f}%)")
print(f"Nero muove:   {black_to_move} ({100*black_to_move/total:.1f}%)")
print(f"Errori:       {errors}")
print(f"Rapporto bianco/nero: {white_to_move/black_to_move:.2f}")
