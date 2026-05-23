"""
filter_lichess.py — Filtra partite Lichess da PGN/ZST e produce un CSV di training.

Input:  file .pgn.zst (o .pgn decompresso) da database.lichess.org
Output: filtered_games.csv con colonne:
          fen, move_uci, outcome
        dove:
          fen      = FEN della posizione prima della mossa
          move_uci = mossa giocata in UCI
          outcome  = risultato dal punto di vista del giocatore che muove
                     +1.0 vittoria, -1.0 sconfitta, 0.0 patta

Filtri applicati:
  - Entrambi i giocatori >= MIN_ELO
  - Time control: solo Rapid e Classical (no Bullet, Blitz, Correspondence)
  - Risultato definito (no partite abbandonate senza risultato)
  - Posizioni con almeno 1 mossa legale

Utilizzo:
  python filter_lichess.py --input lichess_db_2025-12.pgn.zst --output filtered_games.csv
  python filter_lichess.py --input lichess_db_2025-12.pgn --output filtered_games.csv

Dipendenze:
  pip install chess zstandard tqdm pandas
"""

import argparse
import csv
import io
import os
import sys
import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

MIN_ELO         = 2000
MAX_GAMES       = None        # None = nessun limite
MIN_MOVES       = 10          # scarta partite troppo corte (errori/abbandoni)
MAX_MOVES       = 300         # scarta partite anomale
SKIP_OPENINGS   = 5           # salta le prime N mosse (apertura teorica poco informativa)

VALID_TIME_CONTROLS = {
    "rapid",
    "classical",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_elo(headers, color: str) -> int:
    try:
        return int(headers.get(f"{color}Elo", 0))
    except (ValueError, TypeError):
        return 0


def parse_time_control(headers) -> str:
    """Restituisce il tipo di time control Lichess ('bullet','blitz','rapid','classical')."""
    return headers.get("Event", "").lower().split(" ")[1] if " " in headers.get("Event", "") else ""


def parse_outcome(result: str, turn: chess.Color) -> float:
    """
    Converte il risultato della partita in win probability dal punto di vista
    del giocatore che muove.
    turn: chess.WHITE o chess.BLACK
    """
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    elif result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    elif result == "1/2-1/2":
        return 0.0
    return None  # risultato sconosciuto


def open_pgn(path: str):
    """Apre un file PGN o PGN.ZST e restituisce un file-like object."""
    if path.endswith(".zst"):
        print(f"Decompressione ZST: {path}")
        dctx   = zstd.ZstdDecompressor()
        fh     = open(path, "rb")
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        return open(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filtra partite Lichess")
    parser.add_argument("--input",    required=True,  help="File .pgn o .pgn.zst")
    parser.add_argument("--output",   default="filtered_games.csv")
    parser.add_argument("--min-elo",  type=int, default=MIN_ELO)
    parser.add_argument("--max-games", type=int, default=MAX_GAMES)
    parser.add_argument("--skip-openings", type=int, default=SKIP_OPENINGS)
    args = parser.parse_args()

    min_elo       = args.min_elo
    max_games     = args.max_games
    skip_openings = args.skip_openings

    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Min ELO:      {min_elo} (entrambi i giocatori)")
    print(f"Time control: {VALID_TIME_CONTROLS}")
    print(f"Skip opening: prime {skip_openings} mosse")
    print()

    games_read     = 0
    games_accepted = 0
    positions_written = 0

    with open_pgn(args.input) as pgn_file, \
         open(args.output, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["fen", "move_uci", "outcome"])

        pbar = tqdm(desc="Partite lette", unit=" games", dynamic_ncols=True)

        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception:
                continue

            if game is None:
                break

            games_read += 1
            pbar.update(1)

            if max_games and games_accepted >= max_games:
                break

            headers = game.headers

            # Filtro risultato
            result = headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            # Filtro ELO
            white_elo = parse_elo(headers, "White")
            black_elo = parse_elo(headers, "Black")
            if white_elo < min_elo or black_elo < min_elo:
                continue

            # Filtro time control
            event = headers.get("Event", "").lower()
            tc_ok = any(tc in event for tc in VALID_TIME_CONTROLS)
            if not tc_ok:
                continue

            # Estrai mosse
            moves = list(game.mainline_moves())
            if len(moves) < MIN_MOVES or len(moves) > MAX_MOVES:
                continue

            games_accepted += 1

            # Replay partita e scrivi posizioni
            board    = game.board()
            move_idx = 0

            for move in moves:
                # Salta le aperture
                if move_idx < skip_openings:
                    board.push(move)
                    move_idx += 1
                    continue

                # Verifica mossa legale
                if move not in board.legal_moves:
                    board.push(move)
                    move_idx += 1
                    continue

                # Calcola outcome dal punto di vista del giocatore che muove
                outcome = parse_outcome(result, board.turn)
                if outcome is None:
                    board.push(move)
                    move_idx += 1
                    continue

                fen      = board.fen()
                move_uci = move.uci()

                writer.writerow([fen, move_uci, outcome])
                positions_written += 1

                board.push(move)
                move_idx += 1

            pbar.set_postfix({
                "accepted": games_accepted,
                "positions": positions_written,
            })

        pbar.close()

    print(f"\nPartite lette:     {games_read:,}")
    print(f"Partite accettate: {games_accepted:,}")
    print(f"Posizioni scritte: {positions_written:,}")
    print(f"Output:            {args.output}")
    print(f"Dimensione file:   {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
