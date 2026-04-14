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
import time
import numpy as np
import torch
import chess
from typing import Optional
 
 
# ---------------------------------------------------------------------------
# Nodo dell'albero  (board lazy: ricostruita on-demand)
# ---------------------------------------------------------------------------
 
class MCTSNode:
    __slots__ = (
        "parent", "move",       # struttura albero
        "_board",               # cache board (None finche' non serve)
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "is_expanded",
        "is_terminal",
    )
 
    def __init__(
        self,
        board: chess.Board,     # board GIA' nello stato di questo nodo
        prior: float = 1.0,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
    ):
        self._board = board     # per la radice e per i figli espansi
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.is_terminal = board.is_game_over()
 
    @property
    def board(self) -> chess.Board:
        return self._board
 
    @property
    def Q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
 
    def puct_score(self, c_puct: float) -> float:
        assert self.parent is not None
        u = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.Q + u
 
    def best_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))
 
 
# ---------------------------------------------------------------------------
# MCTS con batch expansion
# ---------------------------------------------------------------------------
 
class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        move_vocab: dict[str, int],
        device: torch.device,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 16,
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.move_vocab = move_vocab
        self.vocab_size = max(move_vocab.values()) + 1
 
    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------
 
    def get_best_move(
        self,
        board: chess.Board,
        num_simulations: Optional[int] = None,
        time_limit_s: Optional[float] = None,
        temperature: float = 0.0,
    ) -> chess.Move:
        if num_simulations is None and time_limit_s is None:
            num_simulations = 800
        root = self._run_simulations(board, num_simulations, time_limit_s)
        return self._select_move(root, temperature)
 
    def get_policy_target(
        self,
        board: chess.Board,
        num_simulations: int = 800,
        temperature: float = 1.0,
    ) -> dict[chess.Move, float]:
        root = self._run_simulations(board, num_simulations, None)
        return self._visit_distribution(root, temperature)
 
    # ------------------------------------------------------------------
    # Loop principale
    # ------------------------------------------------------------------
 
    def _run_simulations(
        self,
        board: chess.Board,
        num_simulations: Optional[int],
        time_limit_s: Optional[float],
    ) -> MCTSNode:
 
        root = MCTSNode(board.copy())
        self._expand_batch([root])
        self._add_dirichlet_noise(root)
 
        deadline = (time.monotonic() + time_limit_s) if time_limit_s else None
        sims_done = 1
 
        while True:
            if num_simulations and sims_done >= num_simulations:
                break
            if deadline and time.monotonic() >= deadline:
                break
 
            # Seleziona N foglie con PUCT puro, poi deduplicale
            raw_leaves  = self._select_batch(root)
            unique, _   = self._deduplicate(raw_leaves)
 
            to_expand = [n for n in unique if not n.is_terminal and not n.is_expanded]
            terminals  = [n for n in unique if n.is_terminal or n.is_expanded]
 
            value_map: dict[int, float] = {}
            if to_expand:
                for node, v in zip(to_expand, self._expand_batch(to_expand)):
                    value_map[id(node)] = v
            for node in terminals:
                value_map[id(node)] = self._terminal_value(node)
 
            # Backprop per ogni foglia del batch originale (duplicati inclusi)
            for leaf in raw_leaves:
                self._backpropagate(leaf, value_map.get(id(leaf), 0.0))
 
            sims_done += len(raw_leaves)
 
        return root
 
    # ------------------------------------------------------------------
    # Selezione batch — PUCT puro, no virtual loss
    # ------------------------------------------------------------------
 
    def _select_batch(self, root: MCTSNode) -> list[MCTSNode]:
        leaves = []
        for _ in range(self.batch_size):
            node = root
            while node.is_expanded and not node.is_terminal:
                if not node.children:
                    break
                node = node.best_child(self.c_puct)
            leaves.append(node)
        return leaves
 
    @staticmethod
    def _deduplicate(leaves: list[MCTSNode]) -> tuple[list[MCTSNode], dict[int, int]]:
        seen: dict[int, int] = {}
        unique: list[MCTSNode] = []
        for node in leaves:
            nid = id(node)
            if nid not in seen:
                seen[nid] = len(unique)
                unique.append(node)
        return unique, seen
 
    # ------------------------------------------------------------------
    # Expand batch — una sola forward pass per N nodi unici
    # ------------------------------------------------------------------
 
    def _expand_batch(self, nodes: list[MCTSNode]) -> list[float]:
        valid = [n for n in nodes if not n.is_terminal]
        if not valid:
            return [self._terminal_value(n) for n in nodes]
 
        tensors = [self._board_to_tensor(n.board) for n in valid]
        batch   = torch.cat(tensors, dim=0)                         # (N, 13, 8, 8)
        policy_batch, value_batch = self._infer_batch(batch, valid) # (N,V), (N,)
 
        values = []
        for i, node in enumerate(valid):
            policy_logits = policy_batch[i]
            value_scalar  = value_batch[i].item()
 
            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                node.is_terminal = True
                values.append(self._terminal_value(node))
                continue
 
            priors = self._masked_softmax(policy_logits, legal_moves)
            node.is_expanded = True
 
            # Lazy board: per ogni figlio costruiamo la board con push
            # cosi' non duplichiamo la struttura per ogni mossa legale
            for move in legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(
                    board  = child_board,
                    prior  = priors[move],
                    parent = node,
                    move   = move,
                )
            values.append(value_scalar)
 
        return values
 
    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
 
    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum   += value
            value    = -value
            current  = current.parent
 
    # ------------------------------------------------------------------
    # Selezione mossa finale
    # ------------------------------------------------------------------
 
    def _select_move(self, root: MCTSNode, temperature: float) -> chess.Move:
        if not root.children:
            return list(root.board.legal_moves)[0]
        if temperature == 0.0:
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float64)
        moves  = list(root.children.keys())
        visits = visits ** (1.0 / temperature)
        probs  = visits / visits.sum()
        return moves[np.random.choice(len(moves), p=probs)]
 
    def _visit_distribution(self, root: MCTSNode, temperature: float) -> dict[chess.Move, float]:
        visits = {m: c.visit_count for m, c in root.children.items()}
        total  = sum(v ** (1.0 / temperature) for v in visits.values()) or 1e-8
        return {m: (v ** (1.0 / temperature)) / total for m, v in visits.items()}
 
    # ------------------------------------------------------------------
    # Utilita'
    # ------------------------------------------------------------------
 
    def _terminal_value(self, node: MCTSNode) -> float:
        outcome = node.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner != node.board.turn else -1.0
 
    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Sovrascrivere nella sottoclasse. Restituisce (1, 13, 8, 8) su device."""
        raise NotImplementedError
 
    @torch.no_grad()
    def _infer_batch(
        self,
        batch: torch.Tensor,
        nodes: list[MCTSNode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sovrascrivere nella sottoclasse.
        Restituisce (policy_batch, value_batch) su CPU:
          policy_batch: (N, vocab_size)
          value_batch:  (N,)
        """
        raise NotImplementedError
 
    def _masked_softmax(
        self,
        logits: torch.Tensor,
        legal_moves: list[chess.Move],
    ) -> dict[chess.Move, float]:
        mask        = torch.full((self.vocab_size,), float("-inf"))
        move_indices: dict[chess.Move, int] = {}
        for move in legal_moves:
            uci = move.uci()
            if uci in self.move_vocab:
                idx = self.move_vocab[uci]
                mask[idx] = 0.0
                move_indices[move] = idx
 
        if not move_indices:
            u = 1.0 / len(legal_moves)
            return {m: u for m in legal_moves}
 
        probs  = torch.softmax(logits + mask, dim=0)
        result = {
            m: probs[move_indices[m]].item() if m in move_indices else 1e-8
            for m in legal_moves
        }
        total = sum(result.values())
        return {m: p / total for m, p in result.items()}
 
    def _add_dirichlet_noise(self, root: MCTSNode):
        if not root.children:
            return
        children = list(root.children.values())
        noise    = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps      = self.dirichlet_epsilon
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

class JellyfishMCTS(MCTS):
 
    def __init__(self, model, move_vocab, device, transform, **kwargs):
        super().__init__(model, move_vocab, device, **kwargs)
        self.transform = transform
 
    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        fen           = board.fen()
        legal_ucis    = [str(m) for m in board.legal_moves]
        legal_indices = [self.move_vocab[m] for m in legal_ucis if m in self.move_vocab]
        board_tensor, _, _, _ = self.transform(fen, "", legal_indices, "0")
        return board_tensor.unsqueeze(0).to(self.device)            # (1, 13, 8, 8)
 
    def _make_mask(self, board: chess.Board) -> torch.Tensor:
        """Maschera booleana (1, vocab_size) per le mosse legali."""
        legal_ucis    = [str(m) for m in board.legal_moves]
        legal_indices = [self.move_vocab[m] for m in legal_ucis if m in self.move_vocab]
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for idx in legal_indices:
            mask[idx] = True
        return mask.unsqueeze(0)                                     # (1, vocab_size)
 
    @torch.no_grad()
    def _infer_batch(
        self,
        batch: torch.Tensor,            # (N, 13, 8, 8) gia' su device
        nodes: list[MCTSNode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatena le maschere per-nodo e chiama forward_phase2 una volta sola.
        Restituisce (policy_batch, value_batch) su CPU.
        """
        self.model.eval()
        masks = torch.cat(
            [self._make_mask(n.board) for n in nodes], dim=0
        ).to(self.device)                                            # (N, vocab_size)
 
        policy_batch, value_batch = self.model.forward_phase2(batch, masks)
        return policy_batch.cpu(), value_batch.squeeze(-1).cpu()    # (N,V), (N,)
 