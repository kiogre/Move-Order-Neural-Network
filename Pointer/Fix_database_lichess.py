"""
fix_dataset.py — Analizza e corregge le mosse illegali nel CSV.

Produce:
  1. arrocchi_chess960.csv   — arrocchi in notazione X-FEN (e1h1 → e1g1 ecc.)
  2. mosse_illegali_altre.csv — mosse illegali per altri motivi
  3. fen_zero_mosse.csv       — posizioni senza mosse legali

E in output:
  stockfish_lichess_fixed.csv — CSV corretto, pronto per il training

Utilizzo:
  python fix_dataset.py --input stockfish_lichess_20m.csv --output stockfish_lichess_fixed.csv
"""

import argparse
import chess
import pandas as pd
from tqdm import tqdm

# Mapping arrocco Chess960: (pezzo che muove, casella di destinazione) → mossa UCI standard
# Funziona solo quando il re si trova nella casella di partenza standard

def try_fix_castling(board: chess.Board, move_uci: str) -> str | None:
    """
    Prova a interpretare una mossa illegale come arrocco Chess960.
    
    In Chess960 l'arrocco viene scritto come "re → torre" (es. e1h1).
    Python-chess si aspetta "re → casella destinazione" (es. e1g1).
    
    Ritorna la mossa corretta in UCI se è un arrocco, None altrimenti.
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return None

    from_sq = move.from_square
    to_sq   = move.to_square

    # Il pezzo che muove deve essere un re
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.KING:
        return None

    # La casella di destinazione deve avere una torre dello stesso colore
    target = board.piece_at(to_sq)
    if target is None or target.piece_type != chess.ROOK:
        return None
    if target.color != piece.color:
        return None

    # E' un arrocco Chess960 — trova la mossa corretta tra le mosse legali
    for legal in board.legal_moves:
        if legal.from_square == from_sq and board.is_castling(legal):
            # Verifica che sia il tipo giusto (kingside/queenside)
            king_file  = chess.square_file(from_sq)
            rook_file  = chess.square_file(to_sq)
            dest_file  = chess.square_file(legal.to_square)

            if rook_file > king_file and dest_file > king_file:
                # Entrambi vanno a destra → kingside
                return legal.uci()
            elif rook_file < king_file and dest_file < king_file:
                # Entrambi vanno a sinistra → queenside
                return legal.uci()

    return None


def classify_and_fix(csv_input: str, csv_output: str):
    print(f"Caricamento: {csv_input}")
    df = pd.read_csv(csv_input)
    df = df.dropna(subset=["FEN", "Move", "Evaluation"])
    df = df.reset_index(drop=True)
    print(f"Posizioni totali: {len(df):,}\n")

    rows_ok            = []
    rows_castling      = []   # arrocchi Chess960 — corretti
    rows_illegal_other = []   # illegali per altri motivi
    rows_zero_moves    = []   # zero mosse legali

    for i, row in tqdm(df.iterrows(), total=len(df), unit=" pos"):
        fen      = str(row["FEN"])
        move_uci = str(row["Move"])
        eval_str = str(row["Evaluation"])

        try:
            board      = chess.Board(fen)
            legal_list = list(board.legal_moves)
        except Exception as e:
            rows_illegal_other.append({**row, "error": f"invalid_fen: {e}"})
            continue

        if len(legal_list) == 0:
            rows_zero_moves.append(dict(row))
            continue

        try:
            move = chess.Move.from_uci(move_uci)
        except Exception:
            rows_illegal_other.append({**row, "error": "invalid_uci"})
            continue

        if move in legal_list:
            # Mossa valida — ok
            rows_ok.append(dict(row))
        else:
            # Prova a fixare come arrocco Chess960
            fixed = try_fix_castling(board, move_uci)
            if fixed is not None:
                fixed_move = chess.Move.from_uci(fixed)
                if fixed_move in legal_list:
                    new_row = dict(row)
                    new_row["Move"] = fixed
                    rows_castling.append({**row, "Move_originale": move_uci})
                    rows_ok.append(new_row)
                else:
                    rows_illegal_other.append({**row, "error": "castling_fix_failed"})
            else:
                rows_illegal_other.append({**row, "error": "move_not_legal"})

    # ---------------------------------------------------------------------------
    # Salva i file di diagnostica
    # ---------------------------------------------------------------------------
    pd.DataFrame(rows_castling).to_csv("arrocchi_chess960.csv", index=False)
    pd.DataFrame(rows_illegal_other).to_csv("mosse_illegali_altre.csv", index=False)
    pd.DataFrame(rows_zero_moves).to_csv("fen_zero_mosse.csv", index=False)

    # ---------------------------------------------------------------------------
    # Salva il CSV corretto
    # ---------------------------------------------------------------------------
    df_fixed = pd.DataFrame(rows_ok)
    df_fixed.to_csv(csv_output, index=False)

    print(f"\nRisultati:")
    print(f"  Posizioni OK (inclusi arrocchi fixati): {len(rows_ok):,}")
    print(f"  Arrocchi Chess960 corretti:             {len(rows_castling):,}")
    print(f"  Mosse illegali (droppate):              {len(rows_illegal_other):,}")
    print(f"  Posizioni zero mosse (droppate):        {len(rows_zero_moves):,}")
    print(f"\nOutput:")
    print(f"  {csv_output}               ({len(rows_ok):,} posizioni)")
    print(f"  arrocchi_chess960.csv      ({len(rows_castling):,} righe)")
    print(f"  mosse_illegali_altre.csv   ({len(rows_illegal_other):,} righe)")
    print(f"  fen_zero_mosse.csv         ({len(rows_zero_moves):,} righe)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="stockfish_lichess_60m.csv")
    parser.add_argument("--output", default="stockfish_lichess_60m_fixed.csv")
    args = parser.parse_args()
    classify_and_fix(args.input, args.output)


if __name__ == "__main__":
    main()