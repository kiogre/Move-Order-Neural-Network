"""
mcts.py — AlphaZero-style MCTS integrato con ChessModel (policy + value head).

Dipendenze:
    - python-chess
    - torch
    - numpy
    - Il tuo FullChessModel (o FullChessModelWithAdversarial)
    - generate_all_legal_move_vocab()  →  { "e2e4": 42, ... }
    - fen_to_tensor(fen)               →  np.ndarray (13, 8, 8)

Utilizzo minimo:
    mcts = MCTS(model, move_vocab, device)
    move  = mcts.get_best_move(board, num_simulations=800)
    probs = mcts.get_policy_target(board, num_simulations=800, temperature=1.0)
"""

import math
import numpy as np
import torch
import chess
from typing import Optional


# ---------------------------------------------------------------------------
# Nodo dell'albero
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "board", "parent", "move",      # struttura albero
        "prior",                         # P(s,a) dalla policy head della rete
        "children",                      # {chess.Move: MCTSNode}
        "visit_count",
        "value_sum",
        "is_expanded",
        "is_terminal",
    )

    def __init__(
        self,
        board: chess.Board,
        prior: float = 1.0,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
    ):
        self.board = board
        self.parent = parent
        self.move = move            # mossa che ha portato a questo nodo
        self.prior = prior
        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.is_terminal = board.is_game_over()

    # ------------------------------------------------------------------
    # Q(s, a): valore medio backpropagato
    # ------------------------------------------------------------------
    @property
    def Q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    # ------------------------------------------------------------------
    # PUCT score  (formula AlphaZero)
    # ------------------------------------------------------------------
    def puct_score(self, c_puct: float) -> float:
        assert self.parent is not None, "Il nodo radice non ha uno score PUCT."
        u = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.Q + u

    # ------------------------------------------------------------------
    # Selezione del miglior figlio
    # ------------------------------------------------------------------
    def best_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))


# ---------------------------------------------------------------------------
# MCTS principale
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        move_vocab: dict[str, int],          # output di generate_all_legal_move_vocab()
        device: torch.device,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,        # rumore alla radice (esplorazione)
        dirichlet_epsilon: float = 0.25,     # peso del rumore Dirichlet
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Vocabolario inverso: indice → stringa UCI
        self.move_vocab = move_vocab                   # str -> int
        self.vocab_size = max(move_vocab.values()) + 1

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def get_best_move(
        self,
        board: chess.Board,
        num_simulations: int = 800,
        temperature: float = 0.0,   # 0 = greedy (gioco), 1 = soft (training)
    ) -> chess.Move:
        """
        Esegue `num_simulations` simulazioni e restituisce la mossa migliore.
        """
        root = self._run_simulations(board, num_simulations)
        return self._select_move(root, temperature)

    def get_policy_target(
        self,
        board: chess.Board,
        num_simulations: int = 800,
        temperature: float = 1.0,
    ) -> dict[chess.Move, float]:
        """
        Restituisce la distribuzione di visita (target per il training della policy head).
        { chess.Move: probabilità }
        """
        root = self._run_simulations(board, num_simulations)
        return self._visit_distribution(root, temperature)

    # ------------------------------------------------------------------
    # Loop principale
    # ------------------------------------------------------------------

    def _run_simulations(self, board: chess.Board, num_simulations: int) -> MCTSNode:
        root = MCTSNode(board.copy())

        # Espandi la radice e aggiungi rumore Dirichlet
        value = self._expand(root)
        self._backpropagate(root, value)
        self._add_dirichlet_noise(root)

        for _ in range(num_simulations - 1):
            node = self._select(root)

            if node.is_terminal:
                value = self._terminal_value(node)
            elif not node.is_expanded:
                value = self._expand(node)
            else:
                # Può succedere se il nodo è già stato espanso
                # ma non ha ancora figli (posizione senza mosse legali).
                value = self._terminal_value(node)

            self._backpropagate(node, value)

        return root

    # ------------------------------------------------------------------
    # Select: scende lungo l'albero seguendo PUCT finché non trova un
    #         nodo non espanso (o terminale)
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            node = node.best_child(self.c_puct)
        return node

    # ------------------------------------------------------------------
    # Expand: chiama la rete, crea i figli con le prior
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> float:
        """
        Espande `node` usando la rete per ottenere policy e value.
        Restituisce il value della posizione (dal punto di vista del
        giocatore di turno, normalizzato in [-1, 1]).
        """
        if node.is_terminal:
            return self._terminal_value(node)

        tensor = self._board_to_tensor(node.board)          # (1, 13, 8, 8)
        policy_logits, value = self._infer(tensor)           # policy: (vocab_size,), value: scalar

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.is_terminal = True
            return self._terminal_value(node)

        # Maschera sulle mosse legali
        policy_priors = self._masked_softmax(policy_logits, legal_moves, node.board)

        node.is_expanded = True
        for move in legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(
                board=child_board,
                prior=policy_priors[move],
                parent=node,
                move=move,
            )

        # Il value dalla rete è dal punto di vista del giocatore di turno:
        # se il nodo è del Bianco, value > 0 → buono per il Bianco.
        # La backprop inverte il segno salendo lungo l'albero.
        return float(value)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Risale l'albero aggiornando visit_count e value_sum.
        Il value viene invertito ad ogni livello perché i turni si alternano.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value          # inversione prospettiva
            current = current.parent

    # ------------------------------------------------------------------
    # Selezione della mossa finale
    # ------------------------------------------------------------------

    def _select_move(self, root: MCTSNode, temperature: float) -> chess.Move:
        if not root.children:
            # Fallback: mossa casuale (non dovrebbe mai succedere)
            return list(root.board.legal_moves)[0]

        if temperature == 0.0:
            # Greedy: mossa più visitata
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        # Temperature sampling
        visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float64)
        moves  = list(root.children.keys())
        visits = visits ** (1.0 / temperature)
        probs  = visits / visits.sum()
        idx    = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def _visit_distribution(
        self, root: MCTSNode, temperature: float
    ) -> dict[chess.Move, float]:
        visits = {m: c.visit_count for m, c in root.children.items()}
        total  = sum(v ** (1.0 / temperature) for v in visits.values()) or 1e-8
        return {
            m: (v ** (1.0 / temperature)) / total
            for m, v in visits.items()
        }

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def _terminal_value(self, node: MCTSNode) -> float:
        """
        Restituisce il valore della posizione terminale dal punto di vista
        del giocatore che ha *appena mosso* (i.e., opposto al turno attuale).
        """
        outcome = node.board.outcome()
        if outcome is None:
            return 0.0  # posizione non ancora terminata (safety fallback)
        if outcome.winner is None:
            return 0.0  # patta
        # Se chi ha vinto è il giocatore che ha appena mosso → +1
        # Il turno è ora dell'avversario, quindi il giocatore che ha mosso
        # è l'opposto del turno attuale.
        just_moved = not node.board.turn
        if outcome.winner == just_moved:
            return 1.0
        return -1.0

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Converte una board in tensore (1, 13, 8, 8) sulla device corretta."""
        from ..Representation.Siamese_Autoencoder_Representation import fen_to_tensor   # <-- sostituisci con il tuo import
        arr = fen_to_tensor(board.fen())        # (13, 8, 8) numpy
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _infer(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Chiama il modello e restituisce (policy_logits, value_scalar).

        Adatta il codice se il tuo forward() restituisce in ordine diverso.
        Atteso: model(x) -> (policy_logits, value)
            policy_logits: (1, vocab_size)  — logit grezzi, NON softmax
            value:         (1, 1) o (1,)    — in [-1, 1] (tanh)
        """
        self.model.eval()
        output = self.model(tensor)

        # Adatta qui in base al tuo forward()
        # Esempio: output = (z, reconstructed, policy, value)
        if isinstance(output, tuple):
            policy_logits = output[-2]   # penultimo = policy
            value         = output[-1]   # ultimo    = value
        else:
            raise ValueError(f"Output del modello non riconosciuto: {type(output)}")

        policy_logits = policy_logits.squeeze(0).cpu()   # (vocab_size,)
        value_scalar  = value.squeeze().cpu().item()     # scalar

        return policy_logits, value_scalar

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        legal_moves: list[chess.Move],
        board: chess.Board,
    ) -> dict[chess.Move, float]:
        """
        Applica softmax solo sulle mosse legali, azzerando le illegali.
        Restituisce {move: prior_prob}.
        """
        # Indici delle mosse legali nel vocabolario
        mask = torch.full((self.vocab_size,), float("-inf"))
        move_indices = {}

        for move in legal_moves:
            uci = move.uci()
            if uci in self.move_vocab:
                idx = self.move_vocab[uci]
                mask[idx] = 0.0
                move_indices[move] = idx

        # Se nessuna mossa legale è nel vocabolario (non dovrebbe succedere),
        # usa prior uniforme come fallback sicuro.
        if not move_indices:
            uniform = 1.0 / len(legal_moves)
            return {m: uniform for m in legal_moves}

        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=0)

        result = {}
        for move in legal_moves:
            if move in move_indices:
                result[move] = probs[move_indices[move]].item()
            else:
                result[move] = 1e-8   # prior minima per mosse fuori vocab

        # Rinormalizza (per sicurezza, nel caso ci siano mosse fuori vocab)
        total = sum(result.values())
        return {m: p / total for m, p in result.items()}

    def _add_dirichlet_noise(self, root: MCTSNode):
        """
        Aggiunge rumore Dirichlet alle prior del nodo radice.
        Serve durante il self-play per garantire esplorazione.
        """
        if not root.children:
            return
        children = list(root.children.values())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps = self.dirichlet_epsilon
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * n


# ---------------------------------------------------------------------------
# Esempio d'uso
# ---------------------------------------------------------------------------

'''
if __name__ == "__main__":
    import chess
    from .my_resnet import FullChessModel
    from ..Representation.data_organization_tensor import generate_all_legal_move_vocab

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move_vocab = generate_all_legal_move_vocab()
    model      = FullChessModel(...).to(device)
    # model.load_state_dict(torch.load("checkpoint.pt"))

    mcts  = MCTS(model, move_vocab, device, c_puct=1.5)
    board = chess.Board()

    # -- Gioco: mossa greedy
    best_move = mcts.get_best_move(board, num_simulations=400, temperature=0.0)
    print("Mossa migliore:", best_move)

    # -- Training: distribuzione target per la policy
    policy_target = mcts.get_policy_target(board, num_simulations=400, temperature=1.0)
    print("Policy target (prime 5):", list(policy_target.items())[:5])
'''