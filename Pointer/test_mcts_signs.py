"""
test_mcts_signs.py — Verifica diretta delle convenzioni di segno dell'MCTS.

Tre test indipendenti, dal piu' "isolato" al piu' "integrato":

  PARTE A — _terminal_value su un nodo terminale reale (scacco matto).
            Non richiede ricerca, non richiede una rete addestrata
            (i pesi possono essere random). Verifica solo che
            _terminal_value(node) ritorni il valore corretto dal punto
            di vista di chi muove nel nodo (il lato matto-vato => -1.0).

  PARTE B — puct_score/best_child su nodi costruiti a mano (Q noti).
            Verifica che, dato un nodo padre con due figli di cui uno
            "buono per l'avversario" (Q alto) e uno "cattivo per
            l'avversario" (Q basso), best_child scelga quello cattivo
            per l'avversario (cioe' buono per il genitore).

  PARTE C — Ricerca completa (poche/tante simulazioni) su due posizioni
            di matto in 1 (una col Bianco da muovere, una col Nero da
            muovere, speculari), per vedere se la mossa di matto emerge
            come la migliore (visit_count piu' alto, -Q vicino a +1).

Uso:
    python test_mcts_signs.py
    python test_mcts_signs.py --checkpoint checkpoints_az_v3/last.pt --sims 1000

Se non passi --checkpoint, viene usata una rete con pesi RANDOM: per le
parti A e B va benissimo (non dipendono dalla rete). Per la parte C,
con pesi random la rete non avra' priors/value sensati, quindi e' utile
solo per controllare che, con sims abbastanza alte, la mossa di matto
emerga comunque (perche' il suo _terminal_value=-1/+1 dovrebbe dominare
sul rumore della rete non addestrata). Con un checkpoint vero il test e'
piu' rappresentativo della situazione reale.
"""

import argparse
import chess
import torch

from MLChess import JellyFishPointer, BatchedPointerMCTS

# MCTSNode potrebbe essere esportato da MLChess o stare nel modulo
# pointer_mcts_batched — proviamo entrambi.
try:
    from MLChess import MCTSNode
except ImportError:
    from pointer_mcts_batched import MCTSNode


# ---------------------------------------------------------------------------
# Posizioni di test
# ---------------------------------------------------------------------------

# Matto in 1 col Bianco da muovere: Re1-e8#  (matto di colonna/traversa)
WHITE_MATE_IN_1_FEN  = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
WHITE_MATE_IN_1_MOVE = "e1e8"

# Stessa posizione speculare, Nero da muovere: Re8-e1#
BLACK_MATE_IN_1_FEN  = "4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
BLACK_MATE_IN_1_MOVE = "e8e1"


# ---------------------------------------------------------------------------
# PARTE A — _terminal_value su un nodo terminale reale
# ---------------------------------------------------------------------------

def test_terminal_value(mcts: BatchedPointerMCTS):
    print("\n" + "=" * 70)
    print("PARTE A — _terminal_value su nodi terminali (scacco matto)")
    print("=" * 70)

    for label, fen, mate_move in [
        ("Bianco da' matto",  WHITE_MATE_IN_1_FEN, WHITE_MATE_IN_1_MOVE),
        ("Nero da' matto",    BLACK_MATE_IN_1_FEN, BLACK_MATE_IN_1_MOVE),
    ]:
        board = chess.Board(fen)
        mover_before = "Bianco" if board.turn == chess.WHITE else "Nero"

        board.push(chess.Move.from_uci(mate_move))
        assert board.is_checkmate(), f"{label}: la mossa {mate_move} non e' matto! Controlla il FEN."

        node = MCTSNode(board.copy())
        assert node.is_terminal, "Il nodo dopo lo scacco matto dovrebbe essere is_terminal=True"

        value = mcts._terminal_value(node)

        mated_side = "Bianco" if node.board.turn == chess.WHITE else "Nero"

        print(f"\n[{label}]")
        print(f"  FEN dopo {mate_move}: {board.fen()}")
        print(f"  Lato matto-vato (board.turn dopo il matto): {mated_side}")
        print(f"  _terminal_value(node) = {value}")

        if value == -1.0:
            print("  -> OK: -1.0, cioe' 'pessimo per chi e' matto-vato'. Convenzione corretta.")
        elif value == 1.0:
            print("  -> SEGNO INVERTITO: ritorna +1.0 per il lato matto-vato. "
                  "Questo e' il bug che avevamo discusso, NON ANCORA risolto.")
        else:
            print(f"  -> Valore inatteso ({value}), controlla l'implementazione.")


# ---------------------------------------------------------------------------
# PARTE B — puct_score/best_child su nodi costruiti a mano
# ---------------------------------------------------------------------------

def test_puct_sign_convention(mcts: BatchedPointerMCTS):
    print("\n" + "=" * 70)
    print("PARTE B — puct_score/best_child su Q costruiti a mano")
    print("=" * 70)

    # Posizione qualsiasi solo per avere una board valida da passare a MCTSNode
    base_fen = chess.STARTING_FEN
    parent = MCTSNode(chess.Board(base_fen))
    parent.visit_count = 10  # un parent "ben visitato" per avere U non nulli

    legal = list(parent.board.legal_moves)
    move_good_for_parent = legal[0]   # C2: sara' "buona per il genitore"
    move_bad_for_parent  = legal[1]   # C1: sara' "cattiva per il genitore"

    board_c1 = parent.board.copy(); board_c1.push(move_bad_for_parent)
    board_c2 = parent.board.copy(); board_c2.push(move_good_for_parent)

    c1 = MCTSNode(board_c1, parent=parent, prior=0.5)
    c2 = MCTSNode(board_c2, parent=parent, prior=0.5)

    parent.children = {move_bad_for_parent: c1, move_good_for_parent: c2}

    # C1 = mossa che porta a una posizione ottima per CHI MUOVE in C1
    #      (cioe' l'avversario del genitore) -> Q(C1) = +0.9
    c1.visit_count = 5
    c1.value_sum   = 0.9 * c1.visit_count

    # C2 = mossa che porta a una posizione pessima per CHI MUOVE in C2
    #      (l'avversario) -> Q(C2) = -0.9, quindi ottima per il genitore
    c2.visit_count = 5
    c2.value_sum   = -0.9 * c2.visit_count

    print(f"\nC1 (mossa {move_bad_for_parent.uci()}): Q = {c1.Q:+.2f}  "
          f"(posizione ottima per l'AVVERSARIO del genitore)")
    print(f"C2 (mossa {move_good_for_parent.uci()}): Q = {c2.Q:+.2f}  "
          f"(posizione pessima per l'AVVERSARIO, quindi OTTIMA per il genitore)")

    score_c1 = c1.puct_score(mcts.c_puct)
    score_c2 = c2.puct_score(mcts.c_puct)

    print(f"\npuct_score(C1) = {score_c1:+.3f}")
    print(f"puct_score(C2) = {score_c2:+.3f}")

    best_move, best_child = max(parent.children.items(), key=lambda kv: kv[1].puct_score(mcts.c_puct))

    print(f"\nbest_child(parent) seleziona: {best_move.uci()} "
          f"({'C1 (sbagliato!)' if best_move == move_bad_for_parent else 'C2 (corretto)'})")

    if best_move == move_good_for_parent:
        print("-> OK: viene scelta la mossa buona PER IL GENITORE (Q(C2)=-0.9, cioe' "
              "cattiva per l'avversario). Convenzione di puct_score corretta.")
    else:
        print("-> SEGNO INVERTITO: viene scelta la mossa che e' buona per "
              "l'AVVERSARIO (Q alto), non per il genitore. puct_score non sta "
              "negando Q come dovrebbe.")


# ---------------------------------------------------------------------------
# PARTE C — Ricerca completa su matto in 1
# ---------------------------------------------------------------------------

def run_simple_mcts(mcts: BatchedPointerMCTS, board: chess.Board, num_simulations: int,
                     use_dirichlet: bool = True):
    """
    MCTS sequenziale, singola posizione, usando i building block di
    BatchedPointerMCTS (_expand_nodes_batched, _terminal_value,
    _backpropagate, best_child). Serve solo per diagnostica: e' la
    versione "non batchata" della stessa logica.

    use_dirichlet=True replica get_best_move/_expand_roots_batched, che
    aggiungono rumore di Dirichlet ai prior della radice subito dopo
    l'espansione. Senza questo, mosse con prior molto basso possono
    non venire MAI visitate anche con migliaia di simulazioni (vedi U).
    """
    root = MCTSNode(board.copy())

    if not root.is_terminal:
        mcts._expand_nodes_batched([root])

    if use_dirichlet and not root.is_terminal:
        try:
            mcts._add_dirichlet_noise(root)
        except AttributeError:
            print("  [WARN] _add_dirichlet_noise non trovato: eseguo senza rumore di Dirichlet.")

    root.visit_count = 1

    for _ in range(num_simulations):
        node = root
        while node.is_expanded and not node.is_terminal:
            node = node.best_child(mcts.c_puct, fpu_value=getattr(mcts, "fpu_value", -0.1))

        if node.is_terminal:
            value = mcts._terminal_value(node)
        else:
            values = mcts._expand_nodes_batched([node])
            value = values[0]

        mcts._backpropagate(node, value)

    return root


def test_full_search_mate_in_1(mcts: BatchedPointerMCTS, num_simulations: int, use_dirichlet: bool):
    print("\n" + "=" * 70)
    print(f"PARTE C — Ricerca completa su matto in 1 ({num_simulations} simulazioni, "
          f"dirichlet={'ON' if use_dirichlet else 'OFF'})")
    print("=" * 70)

    for label, fen, mate_move_uci in [
        ("Bianco da muovere, matto in 1", WHITE_MATE_IN_1_FEN, WHITE_MATE_IN_1_MOVE),
        ("Nero da muovere, matto in 1",   BLACK_MATE_IN_1_FEN, BLACK_MATE_IN_1_MOVE),
    ]:
        print(f"\n[{label}]")
        print(f"  FEN: {fen}")
        print(f"  Mossa di matto attesa: {mate_move_uci}")

        board = chess.Board(fen)
        root = run_simple_mcts(mcts, board, num_simulations, use_dirichlet=use_dirichlet)

        mate_move = chess.Move.from_uci(mate_move_uci)

        rows = []
        for move, child in root.children.items():
            q  = child.Q if child.visit_count > 0 else float("nan")
            mq = -q if child.visit_count > 0 else float("nan")
            rows.append((move, child.prior, child.visit_count, q, mq, child.is_terminal))

        rows.sort(key=lambda r: r[2], reverse=True)  # ordina per visit_count

        print(f"\n  {'mossa':>8}  {'prior':>7}  {'visits':>7}  {'Q':>7}  {'-Q':>7}  terminale  ")
        for move, prior, visits, q, mq, is_term in rows[:8]:
            marker = "  <-- MATTO" if move == mate_move else ""
            print(f"  {move.uci():>8}  {prior:7.3f}  {visits:7d}  "
                  f"{q:7.3f}  {mq:7.3f}  {str(is_term):>9}{marker}")

        best_by_visits = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]

        print(f"\n  Mossa con piu' visite: {best_by_visits.uci()}"
              f"  ({'== matto, OK' if best_by_visits == mate_move else '!= matto'})")

        mate_child = root.children.get(mate_move)
        if mate_child is not None and mate_child.visit_count > 0:
            print(f"  -Q della mossa di matto: {-mate_child.Q:+.3f} "
                  f"(atteso vicino a +1.0)")
        elif mate_child is not None:
            print("  La mossa di matto non e' mai stata visitata in questa ricerca.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path a un checkpoint .pt (opzionale, default: pesi random)")
    parser.add_argument("--sims", type=int, default=200,
                         help="Numero di simulazioni per la PARTE C")
    parser.add_argument("--no_dirichlet", action="store_true",
                         help="Disabilita il rumore di Dirichlet alla radice in PARTE C "
                              "(per riprodurre il comportamento 'puro' senza esplorazione forzata)")
    parser.add_argument("--c_puct", type=float, default=2.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = JellyFishPointer().to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Checkpoint caricato: {args.checkpoint}")
    else:
        print("Nessun checkpoint: uso pesi RANDOM (ok per parte A/B, parziale per C).")

    model.eval()

    mcts = BatchedPointerMCTS(model, device, c_puct=args.c_puct)

    test_terminal_value(mcts)
    test_puct_sign_convention(mcts)

    with torch.no_grad():
        test_full_search_mate_in_1(mcts, args.sims, use_dirichlet=not args.no_dirichlet)

    print("\n" + "=" * 70)
    print("Riepilogo: se PARTE A da' -1.0, PARTE B seleziona C2, e in PARTE C")
    print("la mossa di matto ha visit_count massimo e -Q vicino a +1, allora")
    print("le convenzioni di segno sono tutte coerenti e corrette.")
    print("=" * 70)


if __name__ == "__main__":
    main()
