"""
mcts.py — AlphaZero-style MCTS per JellyFishPointer (+ JellyfishMCTS legacy).

Ottimizzazioni rispetto alla versione precedente:
  - Cache board_tensor e legal_moves nel nodo (encode_board/encode_legal_moves
    chiamati una sola volta per nodo, all'espansione)
  - FPU (First Play Urgency): nodi non visitati ricevono fpu_value invece di Q=0,
    migliorando l'esplorazione nelle prime simulazioni
  - Bug fix doppio advance_root: get_best_move NON avanza più _cached_root
    internamente. L'avanzamento è responsabilità esclusiva del training loop
    tramite advance_root(). get_policy_target non tocca _cached_root.
  - Tree reuse robusto: verifica FEN prima di riusare il sottoalbero cached
  - root.visit_count=1 dopo l'espansione di un root fresco: senza questo,
    il primo round di selezione collassa su un solo figlio (vedi
    _get_or_build_root per i dettagli), sprecando simulazioni.
  - batch_size predefinito alzato a 16 (da 8): riduce il numero di forward
    pass per mossa (es. 1000 simulazioni -> ~63 pass invece di 125), che per
    un modello piccolo come questo e' tipicamente il collo di bottiglia
    principale (overhead fisso per pass, non il calcolo). Valori piu' alti
    (24-32) possono dare ulteriore velocita' ma, senza virtual loss, piu'
    leaf nello stesso round possono collassare sullo stesso nodo: se alzi
    ulteriormente batch_size, verifica che mcts_solve_rate (eval_puzzles.py)
    non peggiori.
  - _infer_batch: trasferimenti CPU->GPU dei tensori delle mosse raggruppati
    in una singola chiamata .to(device) invece di una per nodo.

API invariata rispetto alla versione precedente — nessuna modifica al training loop.
"""

import math
import time
import numpy as np
import torch
import chess
from typing import Optional

from .Pointer_model import JellyFishPointer
from ..Representation.pointer_dataset import encode_board, encode_legal_moves
from tqdm import tqdm

MOVE_VECTOR_DIM = 46


# ---------------------------------------------------------------------------
# Nodo dell'albero
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "parent", "move",
        "_board",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "is_expanded",
        "is_terminal",
        # Cache — popolata una volta sola all'espansione del nodo
        "_board_tensor",    # Tensor (13,8,8) su CPU — spostato su device al momento dell'uso
        "_legal_moves",     # list[chess.Move] — calcolata una volta sola
        "virtual_loss",     # int — visite virtuali pendenti (rimosso al backprop)
    )

    def __init__(
        self,
        board:  chess.Board,
        prior:  float = 1.0,
        parent: Optional["MCTSNode"] = None,
        move:   Optional[chess.Move] = None,
    ):
        self._board        = board
        self.parent        = parent
        self.move          = move
        self.prior         = prior
        self.children:     dict[chess.Move, "MCTSNode"] = {}
        self.visit_count   = 0
        self.value_sum     = 0.0
        self.is_expanded   = False
        self.is_terminal   = board.is_game_over(claim_draw=True)
        # Cache non ancora popolata
        self._board_tensor: Optional[torch.Tensor] = None
        self._legal_moves:  Optional[list]         = None
        self.virtual_loss: int = 0

    @property
    def board(self) -> chess.Board:
        return self._board

    @property
    def legal_moves(self) -> list[chess.Move]:
        """Mosse legali con cache — list(board.legal_moves) chiamato una volta sola."""
        if self._legal_moves is None:
            self._legal_moves = list(self._board.legal_moves)
        return self._legal_moves

    def get_board_tensor(self) -> torch.Tensor:
        """
        Tensore board (13, 8, 8) su CPU con cache.
        encode_board() viene chiamato una sola volta per nodo.
        """
        if self._board_tensor is None:
            self._board_tensor = encode_board(self._board.fen())
        return self._board_tensor

    @property
    def Q(self) -> float:
        total = self.visit_count + self.virtual_loss
        if total == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / total

    def puct_score(self, c_puct: float, fpu_value: float = -0.1) -> float:
        """
        PUCT score con FPU (First Play Urgency) e virtual loss.

        Per nodi non ancora visitati, Q viene sostituito da fpu_value (tipicamente
        leggermente negativo) invece di 0.0. Questo riduce l'esplorazione "gratuita"
        di nodi non visitati nelle prime simulazioni, migliorando la qualità della
        selezione quando l'albero è poco sviluppato.

        fpu_value = -0.1 è un default conservativo (AlphaZero usa -0.2 ~ -0.4,
        ma con reti più mature).
        """
        assert self.parent is not None
        total = self.visit_count + self.virtual_loss
        q = fpu_value if total == 0 else -self.Q
        u = (
            c_puct * self.prior
            * math.sqrt(self.parent.visit_count + self.parent.virtual_loss)
            / (1 + total)
        )
        return q + u

    def best_child(self, c_puct: float, fpu_value: float = -0.1) -> "MCTSNode":
        return max(
            self.children.values(),
            key=lambda n: n.puct_score(c_puct, fpu_value),
        )


# ---------------------------------------------------------------------------
# Classe base MCTS (usata da JellyfishMCTS legacy)
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        model:             torch.nn.Module,
        move_vocab:        dict[str, int],
        device:            torch.device,
        c_puct:            float = 1.5,
        fpu_value:         float = -0.1,
        dirichlet_alpha:   float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size:        int   = 128,
    ):
        self.model             = model
        self.device            = device
        self.c_puct            = c_puct
        self.fpu_value         = fpu_value
        self.dirichlet_alpha   = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size        = batch_size
        self.move_vocab        = move_vocab
        self.vocab_size        = max(move_vocab.values()) + 1 if move_vocab else 1

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def get_best_move(
        self,
        board:           chess.Board,
        num_simulations: Optional[int]   = None,
        time_limit_s:    Optional[float] = None,
        temperature:     float           = 0.0,
    ) -> chess.Move:
        if num_simulations is None and time_limit_s is None:
            num_simulations = 800
        root = self._run_simulations(board, num_simulations, time_limit_s)
        root.visit_count = 1
        return self._select_move(root, temperature)

    def get_policy_target(
        self,
        board:           chess.Board,
        num_simulations: int   = 1000,
        temperature:     float = 1.0,
    ) -> dict[chess.Move, float]:
        root = self._run_simulations(board, num_simulations, None)
        root.visit_count = 1
        return self._visit_distribution(root, temperature)

    # ------------------------------------------------------------------
    # Loop principale
    # ------------------------------------------------------------------

    def _run_simulations(
        self,
        board:           chess.Board,
        num_simulations: Optional[int],
        time_limit_s:    Optional[float],
    ) -> MCTSNode:

        root = MCTSNode(board.copy())
        self._expand_batch([root])
        self._add_dirichlet_noise(root)

        deadline  = (time.monotonic() + time_limit_s) if time_limit_s else None
        sims_done = 1

        while True:
            if num_simulations and sims_done >= num_simulations:
                break
            if deadline and time.monotonic() >= deadline:
                break

            raw_leaf_paths = self._select_batch(root)
            raw_leaves = [leaf for leaf, _ in raw_leaf_paths]
            unique, _  = self._deduplicate(raw_leaves)

            to_expand = [n for n in unique if not n.is_terminal and not n.is_expanded]
            terminals  = [n for n in unique if n.is_terminal or n.is_expanded]

            value_map: dict[int, float] = {}
            if to_expand:
                for node, v in zip(to_expand, self._expand_batch(to_expand)):
                    value_map[id(node)] = v
            for node in terminals:
                value_map[id(node)] = self._terminal_value(node)

            for leaf, path in raw_leaf_paths:
                self._backpropagate(leaf, value_map.get(id(leaf), 0.0), path)

            sims_done += len(raw_leaf_paths)

        return root

    # ------------------------------------------------------------------
    # Selezione batch
    # ------------------------------------------------------------------

    def _select_batch(self, root: MCTSNode) -> list[tuple]:
        """
        Seleziona batch_size foglie applicando virtual loss durante la discesa.
        Ritorna lista di (leaf, path) dove path include i nodi visitati (fog inclusa).
        La virtual loss diversifica le traiettorie successive nello stesso step.
        Viene rimossa durante il backprop reale in _backpropagate.
        """
        results = []
        for _ in range(self.batch_size):
            node = root
            path_nodes = []
            while node.is_expanded and not node.is_terminal:
                if not node.children:
                    break
                node = node.best_child(self.c_puct, self.fpu_value)
                path_nodes.append(node)
            for n in path_nodes:
                n.virtual_loss += 1
            results.append((node, path_nodes))
        return results

    @staticmethod
    def _deduplicate(leaves: list[MCTSNode]) -> tuple[list[MCTSNode], dict[int, int]]:
        seen:   dict[int, int]  = {}
        unique: list[MCTSNode]  = []
        for node in leaves:
            nid = id(node)
            if nid not in seen:
                seen[nid] = len(unique)
                unique.append(node)
        return unique, seen

    # ------------------------------------------------------------------
    # Expand batch
    # ------------------------------------------------------------------

    def _expand_batch(self, nodes: list[MCTSNode]) -> list[float]:
        valid = [n for n in nodes if not n.is_terminal]
        if not valid:
            return [self._terminal_value(n) for n in nodes]

        tensors = [self._board_to_tensor(n.board) for n in valid]
        batch   = torch.cat(tensors, dim=0)
        policy_batch, value_batch = self._infer_batch(batch, valid)

        values = []
        for i, node in enumerate(valid):
            policy_logits = policy_batch[i]
            value_scalar  = value_batch[i].item()

            legal_moves = node.legal_moves  # usa la cache
            if not legal_moves:
                node.is_terminal = True
                values.append(self._terminal_value(node))
                continue

            priors = self._masked_softmax(policy_logits, legal_moves)
            node.is_expanded = True

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

    def _backpropagate(self, node: MCTSNode, value: float, path: list):
        """
        Backpropaga value risalendo l'albero da node fino alla radice.
        Rimuove la virtual loss dai nodi in path (applicata in _select_batch).
        """
        for n in path:
            n.virtual_loss -= 1
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum   += value
            value   = -value
            current = current.parent

    # ------------------------------------------------------------------
    # Selezione mossa finale
    # ------------------------------------------------------------------

    def _select_move(self, root: MCTSNode, temperature: float) -> chess.Move:
        if not root.children:
            return root.legal_moves[0]
        if temperature == 0.0:
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        visits = np.array(
            [c.visit_count for c in root.children.values()], dtype=np.float64
        )
        moves = list(root.children.keys())
        visits = visits ** (1.0 / temperature)
        probs  = visits / visits.sum()
        return moves[np.random.choice(len(moves), p=probs)]

    def _visit_distribution(
        self, root: MCTSNode, temperature: float
    ) -> dict[chess.Move, float]:
        visits = {m: c.visit_count for m, c in root.children.items()}
        total  = sum(v ** (1.0 / temperature) for v in visits.values()) or 1e-8
        return {m: (v ** (1.0 / temperature)) / total for m, v in visits.items()}

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def _terminal_value(self, node: MCTSNode) -> float:
        outcome = node.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == node.board.turn else -1.0

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _infer_batch(
        self,
        batch: torch.Tensor,
        nodes: list[MCTSNode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _masked_softmax(
        self,
        logits:      torch.Tensor,
        legal_moves: list[chess.Move],
    ) -> dict[chess.Move, float]:
        mask         = torch.full((self.vocab_size,), float("-inf"))
        move_indices: dict[chess.Move, int] = {}
        for move in legal_moves:
            uci = move.uci()
            if uci in self.move_vocab:
                idx        = self.move_vocab[uci]
                mask[idx]  = 0.0
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
# JellyfishMCTS — legacy (FullChessModel con vocabolario fisso)
# ---------------------------------------------------------------------------

class JellyfishMCTS(MCTS):

    def __init__(self, model, move_vocab, device, transform, **kwargs):
        super().__init__(model, move_vocab, device, **kwargs)
        self.transform = transform

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        fen           = board.fen()
        legal_ucis    = [str(m) for m in board.legal_moves]
        legal_indices = [self.move_vocab[m] for m in legal_ucis if m in self.move_vocab]
        board_tensor, _, _, _ = self.transform(fen, "", legal_indices, "0")
        return board_tensor.unsqueeze(0).to(self.device)

    def _make_mask(self, board: chess.Board) -> torch.Tensor:
        legal_ucis    = [str(m) for m in board.legal_moves]
        legal_indices = [self.move_vocab[m] for m in legal_ucis if m in self.move_vocab]
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for idx in legal_indices:
            mask[idx] = True
        return mask.unsqueeze(0)

    @torch.no_grad()
    def _infer_batch(
        self,
        batch: torch.Tensor,
        nodes: list[MCTSNode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        masks = torch.cat(
            [self._make_mask(n.board) for n in nodes], dim=0
        ).to(self.device)
        policy_batch, value_batch = self.model.forward_phase2(batch, masks)
        return policy_batch.cpu(), value_batch.squeeze(-1).cpu()


# ---------------------------------------------------------------------------
# PointerMCTS — MCTS per JellyFishPointer
# ---------------------------------------------------------------------------

class PointerMCTS(MCTS):
    """
    MCTS per JellyFishPointer.

    Differenze rispetto alla classe base:
    - Nessun vocabolario fisso — i prior vengono direttamente dai probs della
      pointer network sulle mosse legali ordinate.
    - _infer_batch fa una forward pass con padding delle mosse per tutto il batch.
    - Cache board_tensor e legal_moves nel nodo (encode_board/encode_legal_moves
      chiamati una sola volta per nodo).
    - FPU in puct_score (ereditato da MCTSNode).
    - Tree reuse: _cached_root sopravvive tra chiamate successive su posizioni
      consecutive. Il training loop è responsabile di chiamare advance_root()
      dopo ogni mossa — PointerMCTS non avanza mai il root internamente.

    API per il training loop:
        mcts.reset_root()                     # inizio partita
        dist  = mcts.get_policy_target(board, ...)   # prima di giocare la mossa
        move  = scegli_mossa(dist)
        mcts.advance_root(move)               # dopo aver scelto la mossa
        board.push(move)
        # ripeti

    API per la valutazione:
        mcts.reset_root()
        move = mcts.get_best_move(board, ...)
        mcts.advance_root(move)
        board.push(move)
    """

    def __init__(
        self,
        model:             JellyFishPointer,
        device:            torch.device,
        c_puct:            float = 1.5,
        fpu_value:         float = -0.1,
        dirichlet_alpha:   float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size:        int   = 64,
    ):
        super().__init__(
            model             = model,
            move_vocab        = {},
            device            = device,
            c_puct            = c_puct,
            fpu_value         = fpu_value,
            dirichlet_alpha   = dirichlet_alpha,
            dirichlet_epsilon = dirichlet_epsilon,
            batch_size        = batch_size,
        )
        self.vocab_size    = 1
        self.model         = model
        self._cached_root: Optional[MCTSNode] = None

    # ------------------------------------------------------------------
    # Tree reuse — API pubblica
    # ------------------------------------------------------------------

    def advance_root(self, move: chess.Move):
        """
        Avanza il root cached al figlio corrispondente alla mossa giocata.

        Da chiamare nel training loop DOPO aver scelto la mossa e PRIMA di
        board.push(move). Non fare board.push prima di chiamare questo metodo
        — la verifica FEN usa la board del nodo figlio (post-push) per
        confermare che il sottoalbero corrisponda alla posizione attuale.

        Se il figlio non esiste nell'albero (mossa non esplorata durante le
        simulazioni, o root non inizializzato), resetta silenziosamente.
        """
        if self._cached_root is not None and move in self._cached_root.children:
            self._cached_root = self._cached_root.children[move]
            self._cached_root.parent = None  # libera il riferimento al vecchio albero
        else:
            self._cached_root = None

    def reset_root(self):
        """Resetta il tree reuse. Chiamare all'inizio di ogni nuova partita."""
        self._cached_root = None

    # ------------------------------------------------------------------
    # Interfaccia pubblica — sovrascrive la classe base
    #
    # CONTRATTO IMPORTANTE:
    #   - get_best_move e get_policy_target NON modificano _cached_root.
    #   - L'unico modo per avanzare il root è chiamare advance_root(move).
    #   - Questo elimina il bug del doppio avanzamento della versione precedente.
    # ------------------------------------------------------------------

    def get_best_move(
        self,
        board:           chess.Board,
        num_simulations: int   = 100,
        time_limit_s:    Optional[float] = None,
        temperature:     float = 0.0,
        add_noise:       bool  = False,
    ) -> chess.Move:
        """
        Restituisce la mossa migliore secondo MCTS.
        NON avanza _cached_root — chiamare advance_root(move) dopo.

        add_noise: se True, applica rumore Dirichlet ai prior del root
        (utile per generare diversita' durante il self-play). Default
        False: per il gioco vero (es. partite Lichess) il rumore non va
        applicato, altrimenti diluisce i prior anche su tattiche nette
        (es. matti in 1) ad ogni mossa.
        """
        root = self._get_or_build_root(board, add_noise=add_noise)
        root = self._run_simulations_from(root, num_simulations, time_limit_s)
        return self._select_move(root, temperature)

    def get_policy_target(
        self,
        board:           chess.Board,
        num_simulations: int   = 100,
        temperature:     float = 1.0,
        add_noise:       bool  = True,
    ) -> dict[chess.Move, float]:
        """
        Restituisce la distribuzione di visita MCTS come policy target.
        NON avanza _cached_root — chiamare advance_root(move) dopo.

        add_noise: default True qui perche' get_policy_target e' tipicamente
        usato nel training loop (self-play), dove il rumore Dirichlet serve
        a diversificare i dati. Passare add_noise=False per valutazioni
        "pulite" fuori dal training.
        """
        root = self._get_or_build_root(board, add_noise=add_noise)
        root = self._run_simulations_from(root, num_simulations, None)
        return self._visit_distribution(root, temperature)

    # ------------------------------------------------------------------
    # Root management interno
    # ------------------------------------------------------------------

    def _get_or_build_root(self, board: chess.Board, add_noise: bool = False) -> MCTSNode:
        """
        Restituisce il root cached se valido per questa posizione,
        altrimenti costruisce un nuovo root da zero.

        Validità: il root cached è valido se la sua board ha la stessa FEN
        della board corrente (ignora il contatore delle mosse a metà mossa
        per robustezza con posizioni curriculum che possono avere halfmove
        clock diverso).

        Il root (cached o nuovo) viene salvato in self._cached_root, cosi'
        che advance_root() possa effettivamente riusare il sottoalbero dopo
        la mossa scelta — prima questa assegnazione mancava nel ramo "root
        nuovo", quindi _cached_root restava sempre None e il tree reuse non
        scattava mai (ogni mossa ripartiva da una ricerca completamente
        fresca: corretto, ma piu' lento del necessario).
        """
        if self._cached_root is not None:
            # Confronto FEN senza halfmove clock e fullmove number per robustezza
            cached_fen  = " ".join(self._cached_root.board.fen().split()[:4])
            current_fen = " ".join(board.fen().split()[:4])
            if cached_fen == current_fen:
                if add_noise:
                    self._add_dirichlet_noise(self._cached_root)
                return self._cached_root

        # Root non valido o non presente: costruisci da zero
        root = MCTSNode(board.copy())
        self._expand_batch([root])
        if add_noise:
            self._add_dirichlet_noise(root)
        # visit_count=1 (non 0): senza questo, nel primo round di
        # _select_batch tutti i figli hanno U=c_puct*prior*sqrt(0)=0,
        # quindi tutti i puct_score sono uguali (fpu_value) e best_child
        # seleziona sempre lo stesso primo figlio per tutti i leaf del
        # round, sprecando batch_size simulazioni senza differenziare
        # per prior. Stesso fix di BatchedPointerMCTS._expand_roots_batched.
        root.visit_count = 1
        self._cached_root = root
        return root

    # ------------------------------------------------------------------
    # Loop simulazioni da root già inizializzato
    # ------------------------------------------------------------------

    def _run_simulations_from(
        self,
        root:            MCTSNode,
        num_simulations: Optional[int],
        time_limit_s:    Optional[float],
    ) -> MCTSNode:
        deadline  = (time.monotonic() + time_limit_s) if time_limit_s else None
        sims_done = 0

        while True:
            if num_simulations is not None and sims_done >= num_simulations:
                break
            if deadline and time.monotonic() >= deadline:
                break

            raw_leaf_paths = self._select_batch(root)
            raw_leaves = [leaf for leaf, _ in raw_leaf_paths]
            unique, _  = self._deduplicate(raw_leaves)

            to_expand = [n for n in unique if not n.is_terminal and not n.is_expanded]
            terminals  = [n for n in unique if n.is_terminal or n.is_expanded]

            value_map: dict[int, float] = {}
            if to_expand:
                for node, v in zip(to_expand, self._expand_batch(to_expand)):
                    value_map[id(node)] = v
            for node in terminals:
                value_map[id(node)] = self._terminal_value(node)

            for leaf, path in raw_leaf_paths:
                self._backpropagate(leaf, value_map.get(id(leaf), 0.0), path)

            sims_done += len(raw_leaf_paths)

        return root

    # ------------------------------------------------------------------
    # Codifica board con cache del nodo
    # ------------------------------------------------------------------

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        # Usato solo dalla classe base — in PointerMCTS lo ignoriamo
        # perché _expand_batch è completamente sovrascritto
        return encode_board(board.fen()).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Batch inference — una forward pass per tutti i nodi
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_batch(
        self,
        batch: torch.Tensor,    # ignorato — ricostruiamo internamente con cache
        nodes: list[MCTSNode],
    ) -> tuple[list[dict], torch.Tensor]:
        """
        Restituisce:
            policy_batch : lista di N dict {chess.Move: float}
            value_batch  : Tensor (N,) su CPU
        """
        self.model.eval()

        board_tensors = []
        moves_tensors = []
        move_lists    = []
        n_moves_per   = []

        for node in nodes:
            legal_moves = node.legal_moves          # cache — nessuna ricalcolo
            move_lists.append(legal_moves)
            n_moves_per.append(len(legal_moves))
            board_tensors.append(node.get_board_tensor())  # cache — nessuna ricalcolo
            moves_tensors.append(encode_legal_moves(node.board))  # (n_i, 46)

        max_n    = max(n_moves_per)
        B        = len(nodes)
        boards_t = torch.stack(board_tensors).to(self.device)   # (B, 13, 8, 8)

        # Costruisci su CPU e trasferisci una sola volta — evita B trasferimenti
        # H2D separati (uno per nodo) in favore di un singolo trasferimento.
        moves_padded_cpu = torch.zeros(B, max_n, MOVE_VECTOR_DIM)
        move_mask_cpu    = torch.zeros(B, max_n, dtype=torch.bool)

        for i, m in enumerate(moves_tensors):
            n = m.shape[0]
            moves_padded_cpu[i, :n] = m
            move_mask_cpu[i, :n]    = True

        moves_padded = moves_padded_cpu.to(self.device)
        move_mask    = move_mask_cpu.to(self.device)

        _, probs, value_pred = self.model(boards_t, moves_padded, move_mask)
        # probs: (B, max_n) già normalizzati, value_pred: (B, 1)

        policy_batch = []
        for i, legal_moves in enumerate(move_lists):
            n     = n_moves_per[i]
            p     = probs[i, :n].cpu()
            prior = {move: p[j].item() for j, move in enumerate(legal_moves)}
            policy_batch.append(prior)

        return policy_batch, value_pred.squeeze(-1).cpu()

    # ------------------------------------------------------------------
    # Sovrascrittura _expand_batch per usare il nuovo formato di _infer_batch
    # ------------------------------------------------------------------

    def _expand_batch(self, nodes: list[MCTSNode]) -> list[float]:
        valid = [n for n in nodes if not n.is_terminal]
        if not valid:
            return [self._terminal_value(n) for n in nodes]

        dummy  = torch.zeros(len(valid), 13, 8, 8, device=self.device)
        policy_batch, value_batch = self._infer_batch(dummy, valid)

        values = []
        for i, node in enumerate(valid):
            prior_dict   = policy_batch[i]
            value_scalar = value_batch[i].item()

            legal_moves = node.legal_moves  # cache
            if not legal_moves:
                node.is_terminal = True
                values.append(self._terminal_value(node))
                continue

            node.is_expanded = True

            for move in legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(
                    board  = child_board,
                    prior  = prior_dict.get(move, 1e-8),
                    parent = node,
                    move   = move,
                )
            values.append(value_scalar)

        return values

    # ------------------------------------------------------------------
    # _masked_softmax non pertinente per PointerMCTS
    # ------------------------------------------------------------------

    def _masked_softmax(self, logits, legal_moves):
        raise NotImplementedError(
            "PointerMCTS non usa _masked_softmax — i prior vengono da _infer_batch."
        )