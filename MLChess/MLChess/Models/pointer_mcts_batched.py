"""
pointer_mcts_batched.py — MCTS batched per JellyFishPointer.

Invece di fare MCTS per una partita alla volta, gestisce N partite in parallelo
e raccoglie le foglie di tutti gli alberi in un unico batch per la forward pass.
Questo satura la GPU con batch grandi invece di tante forward pass piccole.

Ottimizzazioni rispetto alla versione precedente:
  - Tree reuse: dopo ogni mossa il sottoalbero viene riusato invece di essere
    ricostruito. Il risparmio è proporzionale a visit_count del figlio scelto.
  - Cache board_tensor e legal_moves nel nodo (encode_board chiamato una volta
    per nodo, non ad ogni visita).
  - Deduplicazione foglie in play_games_batched: nel loop principale le foglie
    di tutti gli alberi vengono deduplicate prima della forward pass.
  - FPU (First Play Urgency): nodi non visitati usano fpu_value invece di Q=0.
  - Bug fix sims_remaining: il contatore viene decrementato per step, non per
    foglia — era moltiplicato per leaves_per_tree nella versione precedente.
  - Bug fix frozen move + tree reuse: dopo la mossa del frozen, il root del
    main viene avanzato al figlio corrispondente se presente nell'albero,
    invece di essere sempre ricostruito da zero.
  - Frozen move greedy senza espansione root: il frozen non richiede MCTS,
    usa la forward pass diretta con argmax sui probs.

Utilizzo in train_alphazero_batched.py:
    mcts = BatchedPointerMCTS(model, device)
    all_steps, terminals = mcts.play_games_batched(n_games=64, num_simulations=50)
"""

import math
import numpy as np
import torch
import chess
from typing import Optional

from .Pointer_model import JellyFishPointer
from ..Representation.pointer_dataset import encode_board, encode_legal_moves

MOVE_VECTOR_DIM = 46


# ---------------------------------------------------------------------------
# Nodo MCTS — con cache board_tensor e legal_moves
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "parent", "move", "_board", "prior",
        "children", "visit_count", "value_sum",
        "is_expanded", "is_terminal",
        "_board_tensor", "_legal_moves",
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
        self._board_tensor: Optional[torch.Tensor] = None
        self._legal_moves:  Optional[list]         = None

    @property
    def board(self) -> chess.Board:
        return self._board

    @property
    def legal_moves(self) -> list[chess.Move]:
        if self._legal_moves is None:
            self._legal_moves = list(self._board.legal_moves)
        return self._legal_moves

    def get_board_tensor(self) -> torch.Tensor:
        if self._board_tensor is None:
            self._board_tensor = encode_board(self._board.fen())
        return self._board_tensor

    @property
    def Q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def puct_score(self, c_puct: float, fpu_value: float = -0.1) -> float:
        assert self.parent is not None
        q = fpu_value if self.visit_count == 0 else self.Q
        u = (
            c_puct * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return q + u

    def best_child(self, c_puct: float, fpu_value: float = -0.1) -> "MCTSNode":
        return max(
            self.children.values(),
            key=lambda n: n.puct_score(c_puct, fpu_value),
        )


# ---------------------------------------------------------------------------
# Stato di una singola partita in corso
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, game_idx: int, main_is_white: bool, start_fen: Optional[str] = None):
        self.game_idx      = game_idx
        self.main_is_white = main_is_white
        self.board         = chess.Board(start_fen) if start_fen else chess.Board()
        self.steps: list[dict] = []
        self.move_num      = 0
        self.done          = False
        self.result        = None

        # Albero MCTS corrente — sopravvive tra mosse (tree reuse)
        self.root: Optional[MCTSNode] = None


# ---------------------------------------------------------------------------
# BatchedPointerMCTS
# ---------------------------------------------------------------------------

class BatchedPointerMCTS:
    """
    MCTS batched per JellyFishPointer.

    Gestisce N partite in parallelo raccogliendo foglie da tutti gli alberi
    in un unico batch per la forward pass GPU.
    """

    def __init__(
        self,
        model:             JellyFishPointer,
        device:            torch.device,
        c_puct:            float = 1.5,
        fpu_value:         float = -0.1,
        dirichlet_alpha:   float = 0.3,
        dirichlet_epsilon: float = 0.25,
        leaves_per_tree:   int   = 4,
    ):
        self.model             = model
        self.device            = device
        self.c_puct            = c_puct
        self.fpu_value         = fpu_value
        self.dirichlet_alpha   = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.leaves_per_tree   = leaves_per_tree

    # ------------------------------------------------------------------
    # Interfaccia pubblica: singola posizione (valutazione greedy)
    # ------------------------------------------------------------------

    def get_best_move(
        self,
        board:           chess.Board,
        num_simulations: int   = 100,
        temperature:     float = 0.0,
    ) -> chess.Move:
        root = MCTSNode(board.copy())
        self._expand_nodes_batched([root])
        self._add_dirichlet_noise(root)
        root = self._run_simulations_single(root, num_simulations)
        return self._select_move(root, temperature)

    def get_policy_target(
        self,
        board:           chess.Board,
        num_simulations: int   = 100,
        temperature:     float = 1.0,
    ) -> dict[chess.Move, float]:
        root = MCTSNode(board.copy())
        self._expand_nodes_batched([root])
        self._add_dirichlet_noise(root)
        root = self._run_simulations_single(root, num_simulations)
        return self._visit_distribution(root, temperature)

    # ------------------------------------------------------------------
    # Interfaccia pubblica: N partite in parallelo (self-play training)
    # ------------------------------------------------------------------

    def play_games_batched(
        self,
        n_games:         int,
        num_simulations: int   = 100,
        temp_high:       float = 1.0,
        temp_low:        float = 0.1,
        temp_threshold:  int   = 30,
        max_moves:       int   = 300,
        start_fens:      Optional[list[Optional[str]]] = None,
    ) -> tuple[list[list[dict]], list[float]]:
        """
        Gioca n_games partite in parallelo (main model vs se stesso).

        Args:
            start_fens: lista opzionale di FEN di partenza (len == n_games).
                        None per posizione iniziale standard.

        Returns:
            all_steps   : lista di n_games liste di step per il replay buffer
            terminals   : lista di n_games reward terminali (+1 / 0 / -1)
        """
        self.model.eval()
        self._current_max_moves = max_moves  # usato da _make_frozen_moves_batched

        if start_fens is None:
            start_fens = [None] * n_games

        games = [
            GameState(i, main_is_white=(i % 2 == 0), start_fen=start_fens[i])
            for i in range(n_games)
        ]

        # Espandi le radici di tutte le partite in un unico batch
        self._expand_roots_batched(games)


        # Loop principale
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break

            # ---- Mosse del frozen (greedy, senza MCTS) ----
            # Il frozen gioca tutte le sue mosse prima di raccogliere foglie
            # per il main, così il batch successivo è tutto del main.
            frozen_active = [
                g for g in active
                if not g.done
                and not g.board.is_game_over()
                and g.move_num < max_moves
                and (g.board.turn == chess.WHITE) != g.main_is_white
            ]
            if frozen_active:
                self._make_frozen_moves_batched(frozen_active)
                # Ricalcola active dopo le mosse frozen
                active = [g for g in games if not g.done]

            # Controlla terminazione
            for g in active:
                if g.board.is_game_over() or g.move_num >= max_moves:
                    self._finalize_game(g)

            # Partite dove è il turno del main
            main_active = [
                g for g in games
                if not g.done
                and g.root is not None
                and (g.board.turn == chess.WHITE) == g.main_is_white
            ]
            if not main_active:
                continue

            # ---- Simulazioni MCTS per il main ----
            # Ogni partita ha bisogno di num_simulations simulazioni.
            # Contiamo per step (un passo = leaves_per_tree foglie per albero).
            steps_needed = {g.game_idx: math.ceil(num_simulations / self.leaves_per_tree)
                            for g in main_active}

            while True:
                current = [g for g in main_active
                           if not g.done and steps_needed[g.game_idx] > 0]
                if not current:
                    break

                # Raccogli foglie da tutti gli alberi attivi
                all_pairs: list[tuple[GameState, MCTSNode]] = []
                for g in current:
                    leaves = self._select_leaves(g.root)
                    for leaf in leaves:
                        all_pairs.append((g, leaf))

                # Deduplica per id del nodo (cross-tree: raro ma possibile
                # con transposition table futura; intra-tree: comune)
                seen_ids: dict[int, int] = {}
                unique_pairs: list[tuple[GameState, MCTSNode]] = []
                for pair in all_pairs:
                    nid = id(pair[1])
                    if nid not in seen_ids:
                        seen_ids[nid] = len(unique_pairs)
                        unique_pairs.append(pair)

                to_expand = [(g, n) for g, n in unique_pairs
                             if not n.is_terminal and not n.is_expanded]
                terminals  = [(g, n) for g, n in unique_pairs
                              if n.is_terminal or n.is_expanded]

                value_map: dict[int, float] = {}
                if to_expand:
                    nodes_only = [n for _, n in to_expand]
                    values     = self._expand_nodes_batched(nodes_only)
                    for (_, n), v in zip(to_expand, values):
                        value_map[id(n)] = v
                for _, n in terminals:
                    value_map[id(n)] = self._terminal_value(n)

                # Backprop su tutte le foglie originali (inclusi duplicati)
                for g, leaf in all_pairs:
                    v = value_map.get(id(leaf), 0.0)
                    self._backpropagate(leaf, v)

                # Decrementa contatore per partita (uno step = un giro di foglie)
                for g in current:
                    steps_needed[g.game_idx] -= 1

            # ---- Scegli mossa per ogni partita del main ----
            for g in main_active:
                if g.done or g.board.is_game_over() or g.move_num >= max_moves:
                    self._finalize_game(g)
                    continue

                temp = temp_high if g.move_num < temp_threshold else temp_low
                move = self._select_move(g.root, temp)

                # Salva step per il replay buffer
                legal_moves_list = g.root.legal_moves
                legal_moves_t    = encode_legal_moves(g.board)
                policy_target    = self._visit_distribution(g.root, temp)

                target_vec = torch.zeros(len(legal_moves_list))
                for j, m in enumerate(legal_moves_list):
                    target_vec[j] = policy_target.get(m, 0.0)
                s = target_vec.sum()
                if s > 0:
                    target_vec = target_vec / s

                g.steps.append({
                    "board_fen":     g.board.fen(),
                    "legal_moves":   legal_moves_t,
                    "policy_target": target_vec,
                    "value_target":  None,  # riempito dopo
                })

                # Tree reuse: avanza il root al figlio della mossa scelta
                if move in g.root.children:
                    g.root = g.root.children[move]
                    g.root.parent = None
                else:
                    g.root = None  # ricostruito alla prossima iterazione

                g.board.push(move)
                g.move_num += 1

                if g.board.is_game_over() or g.move_num >= max_moves:
                    self._finalize_game(g)
                elif g.root is None:
                    # Figlio non esplorato — costruisci root fresco
                    g.root = MCTSNode(g.board.copy())
                    self._expand_nodes_batched([g.root])
                    self._add_dirichlet_noise(g.root)
                    g.root.visit_count = 1

        # Assegna value_target e raccogli risultati
        all_steps      = []
        terminals_list = []
        for g in games:
            terminal = self._game_terminal(g)
            for step in g.steps:
                # chi muoveva in questo step?
                board_turn = chess.Board(step["board_fen"]).turn
                is_main = (board_turn == chess.WHITE) == g.main_is_white
                step["value_target"] = terminal if is_main else -terminal
            all_steps.append(g.steps)
            terminals_list.append(terminal)

        return all_steps, terminals_list

    # ------------------------------------------------------------------
    # Simulazioni per singola posizione (valutazione)
    # ------------------------------------------------------------------

    def _run_simulations_single(
        self,
        root:            MCTSNode,
        num_simulations: int,
    ) -> MCTSNode:
        sims_done = 1  # il root è già espanso
        while sims_done < num_simulations:
            leaves    = self._select_leaves(root)
            unique    = list({id(n): n for n in leaves}.values())

            to_expand = [n for n in unique if not n.is_terminal and not n.is_expanded]
            terminals  = [n for n in unique if n.is_terminal or n.is_expanded]

            value_map: dict[int, float] = {}
            if to_expand:
                for n, v in zip(to_expand, self._expand_nodes_batched(to_expand)):
                    value_map[id(n)] = v
            for n in terminals:
                value_map[id(n)] = self._terminal_value(n)

            for leaf in leaves:
                self._backpropagate(leaf, value_map.get(id(leaf), 0.0))
                sims_done += 1

        return root

    # ------------------------------------------------------------------
    # Espansione radici in batch (inizio partite)
    # ------------------------------------------------------------------

    def _expand_roots_batched(self, games: list[GameState]):
        roots = []
        for g in games:
            g.root = MCTSNode(g.board.copy())
            roots.append(g.root)
        self._expand_nodes_batched(roots)
        for root in roots:
            self._add_dirichlet_noise(root)
            root.visit_count = 1

    # ------------------------------------------------------------------
    # Mosse del frozen — batch greedy senza MCTS
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _make_frozen_moves_batched(self, games: list[GameState]):
        """
        Esegue le mosse greedy del frozen per tutte le partite in un unico batch.
        Il frozen non usa MCTS — sceglie la mossa con prob massima dalla rete.
        Dopo la mossa del frozen, il root del main viene avanzato per il tree reuse.
        """
        board_tensors = []
        moves_tensors = []
        n_moves_per   = []
        valid_games   = []

        for g in games:
            if g.done or g.board.is_game_over() or g.move_num >= self._current_max_moves:
                self._finalize_game(g)
                continue
            legal_moves = list(g.board.legal_moves)
            if not legal_moves:
                self._finalize_game(g)
                continue
            n_moves_per.append(len(legal_moves))
            board_tensors.append(encode_board(g.board.fen()))
            moves_tensors.append(encode_legal_moves(g.board))
            valid_games.append(g)

        if not valid_games:
            return

        max_n    = max(n_moves_per)
        B        = len(valid_games)
        boards_t = torch.stack(board_tensors).to(self.device)

        moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=self.device)
        move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)
        for i, m in enumerate(moves_tensors):
            n = m.shape[0]
            moves_padded[i, :n] = m.to(self.device)
            move_mask[i, :n]    = True

        _, probs, _ = self.model(boards_t, moves_padded, move_mask)
        # probs: (B, max_n)

        for i, g in enumerate(valid_games):
            legal_moves = list(g.board.legal_moves)
            n           = n_moves_per[i]
            move_idx    = probs[i, :n].argmax().item()
            move        = legal_moves[move_idx]

            # Tree reuse sul root del main: avanza al figlio se esiste
            if g.root is not None and move in g.root.children:
                g.root = g.root.children[move]
                g.root.parent = None
            else:
                g.root = None  # sarà ricostruito quando torna il turno del main

            g.board.push(move)
            g.move_num += 1

            if g.board.is_game_over() or g.move_num >= self._current_max_moves:
                self._finalize_game(g)
            elif g.root is None:
                # Root non disponibile: costruisci fresco (avviene raramente
                # con alberi sufficientemente esplorati)
                g.root = MCTSNode(g.board.copy())
                self._expand_nodes_batched([g.root])
                self._add_dirichlet_noise(g.root)
                g.root.visit_count = 1

    # ------------------------------------------------------------------
    # Selezione foglie da un albero
    # ------------------------------------------------------------------

    def _select_leaves(self, root: MCTSNode) -> list[MCTSNode]:
        leaves = []
        for _ in range(self.leaves_per_tree):
            node = root
            while node.is_expanded and not node.is_terminal:
                if not node.children:
                    break
                node = node.best_child(self.c_puct, self.fpu_value)
            leaves.append(node)
        return leaves

    # ------------------------------------------------------------------
    # Espansione batch — forward pass unica per N nodi unici
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _expand_nodes_batched(self, nodes: list[MCTSNode]) -> list[float]:
        valid = [n for n in nodes if not n.is_terminal]
        if not valid:
            return [self._terminal_value(n) for n in nodes]

        board_tensors = []
        moves_tensors = []
        move_lists    = []
        n_moves_per   = []

        for node in valid:
            legal_moves = node.legal_moves          # cache
            move_lists.append(legal_moves)
            n_moves_per.append(len(legal_moves))
            board_tensors.append(node.get_board_tensor())   # cache
            moves_tensors.append(encode_legal_moves(node.board))

        max_n    = max(n_moves_per)
        B        = len(valid)
        boards_t = torch.stack(board_tensors).to(self.device)

        moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=self.device)
        move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)

        for i, m in enumerate(moves_tensors):
            n = m.shape[0]
            moves_padded[i, :n] = m.to(self.device)
            move_mask[i, :n]    = True

        _, probs, value_pred = self.model(boards_t, moves_padded, move_mask)
        # probs: (B, max_n), value_pred: (B, 1)

        values = []
        for i, node in enumerate(valid):
            legal_moves = move_lists[i]
            n           = n_moves_per[i]

            if not legal_moves:
                node.is_terminal = True
                values.append(self._terminal_value(node))
                continue

            p            = probs[i, :n].cpu()
            node.is_expanded = True

            for j, move in enumerate(legal_moves):
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(
                    board  = child_board,
                    prior  = p[j].item(),
                    parent = node,
                    move   = move,
                )
            values.append(value_pred[i, 0].item())

        return values

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float):
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
        # Temperatura <= 0.05: argmax diretto per evitare overflow in v^(1/temp)
        if temperature <= 0.05:
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        visits = np.array(
            [c.visit_count for c in root.children.values()], dtype=np.float64
        )
        moves  = list(root.children.keys())
        visits = visits ** (1.0 / temperature)
        probs  = visits / visits.sum()
        return moves[np.random.choice(len(moves), p=probs)]

    def _visit_distribution(
        self, root: MCTSNode, temperature: float
    ) -> dict[chess.Move, float]:
        moves  = list(root.children.keys())
        counts = [root.children[m].visit_count for m in moves]
        # Temperatura <= 0.05: one-hot sul massimo per evitare overflow in v^(1/temp)
        if temperature <= 0.05:
            best = int(np.argmax(counts))
            return {m: (1.0 if i == best else 0.0) for i, m in enumerate(moves)}
        visits = {m: c for m, c in zip(moves, counts)}
        total  = sum(v ** (1.0 / temperature) for v in visits.values()) or 1e-8
        return {m: (v ** (1.0 / temperature)) / total for m, v in visits.items()}

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def _terminal_value(self, node: MCTSNode) -> float:
        outcome = node.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner != node.board.turn else -1.0

    def _finalize_game(self, g: GameState):
        if not g.done:
            g.done   = True
            g.result = g.board.result()

    def _game_terminal(self, g: GameState) -> float:
        result = g.result or g.board.result()
        if result == "1-0":
            return  1.0 if g.main_is_white else -1.0
        elif result == "0-1":
            return -1.0 if g.main_is_white else  1.0
        return 0.0

    def _add_dirichlet_noise(self, root: MCTSNode):
        if not root.children:
            return
        children = list(root.children.values())
        noise    = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps      = self.dirichlet_epsilon
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * n