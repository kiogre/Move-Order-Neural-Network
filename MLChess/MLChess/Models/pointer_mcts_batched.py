"""
pointer_mcts_batched.py — MCTS batched per JellyFishPointer (self-play simmetrico).

Invece di fare MCTS per una partita alla volta, gestisce N partite in parallelo
e raccoglie le foglie di tutti gli alberi in un unico batch per la forward pass.
Questo satura la GPU con batch grandi invece di tante forward pass piccole.

CAMBIAMENTO RISPETTO ALLA VERSIONE PRECEDENTE — SELF-PLAY SIMMETRICO:
  In precedenza, ogni partita era "main" (gioca con MCTS completo) vs "frozen"
  (stessa rete, mossa greedy senza ricerca, da self.model). Questo produceva
  partite strutturalmente sbilanciate (MCTS batte quasi sempre la policy nuda
  della stessa rete) e un avversario non realistico.

  Ora ENTRAMBI i lati giocano con MCTS completo (stessa rete), ed entrambi i
  lati vengono registrati come step per il replay buffer. Questo:
    - raddoppia (circa) gli step utili per partita a parità di partite;
    - rende le partite competitive per costruzione (stessa forza sui due lati);
    - allena "main" contro un vero avversario con ricerca, non un fantoccio.

  Costo: il self-play richiede circa il doppio di simulazioni MCTS per partita
  (entrambi i lati fanno num_simulations simulazioni ad ogni mossa, invece di
  uno solo). Se il tempo per epoca cresce troppo, riduci GAMES_PER_EPOCH.

  NOTA sui risultati: play_games_batched ora ritorna i terminal value DAL
  PUNTO DI VISTA DEL BIANCO (+1 = vince il Bianco, -1 = vince il Nero,
  0 = pareggio). Prima erano dal punto di vista di "main". Questo permette
  di monitorare direttamente eventuali squilibri bianco/nero nelle partite
  di self-play.

Ottimizzazioni mantenute dalla versione precedente:
  - Tree reuse: dopo ogni mossa il sottoalbero viene riusato invece di essere
    ricostruito. Il risparmio è proporzionale a visit_count del figlio scelto.
  - Cache board_tensor e legal_moves nel nodo (encode_board chiamato una volta
    per nodo, non ad ogni visita).
  - Deduplicazione foglie in play_games_batched: nel loop principale le foglie
    di tutti gli alberi vengono deduplicate prima della forward pass.
  - FPU (First Play Urgency): nodi non visitati usano fpu_value invece di Q=0.
  - puct_score e _terminal_value: convenzione di segno corretta (Q del figlio
    e' dal punto di vista di chi muove nel figlio; per la selezione al
    genitore va negato).

Utilizzo in train_alphazero_v3.py:
    mcts = BatchedPointerMCTS(model, device)
    all_steps, white_results = mcts.play_games_batched(n_games=64, num_simulations=150)
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
        "virtual_loss",
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
        self.virtual_loss: int = 0

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
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual loss penalizza come se le visite pendenti fossero tutte -1
        return (self.value_sum - self.virtual_loss) / total_visits

    def puct_score(self, c_puct: float, fpu_value: float = -0.1) -> float:
        assert self.parent is not None
        total_visits = self.visit_count + self.virtual_loss
        q = fpu_value if total_visits == 0 else -self.Q
        u = (
            c_puct * self.prior
            * math.sqrt(self.parent.visit_count + self.parent.virtual_loss)
            / (1 + total_visits)
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
    def __init__(self, game_idx: int, start_fen: Optional[str] = None):
        self.game_idx = game_idx
        self.board    = chess.Board(start_fen) if start_fen else chess.Board()
        self.steps: list[dict] = []
        self.move_num = 0
        self.done     = False
        self.result   = None

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
        leaves_per_tree:   int   = 32,
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
        root.visit_count = 1
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
        root.visit_count = 1
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
        Gioca n_games partite in parallelo, self-play simmetrico (stessa rete
        con MCTS completo su entrambi i lati).

        Args:
            start_fens: lista opzionale di FEN di partenza (len == n_games).
                        None per posizione iniziale standard.

        Returns:
            all_steps     : lista di n_games liste di step per il replay buffer
                             (step di ENTRAMBI i lati)
            white_results : lista di n_games esiti dal punto di vista del
                             Bianco (+1 / 0 / -1)
        """
        self.model.eval()

        if start_fens is None:
            start_fens = [None] * n_games

        games = [
            GameState(i, start_fen=start_fens[i])
            for i in range(n_games)
        ]

        # Espandi le radici di tutte le partite in un unico batch
        self._expand_roots_batched(games)

        # Loop principale
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break

            # Controlla terminazione (es. posizioni curriculum già finite)
            for g in active:
                if g.board.is_game_over() or g.move_num >= max_moves:
                    self._finalize_game(g)

            active = [g for g in games if not g.done]
            if not active:
                break

            # ---- Simulazioni MCTS per tutte le partite attive ----
            # Ogni partita ha bisogno di num_simulations simulazioni.
            # Contiamo per step (un passo = leaves_per_tree foglie per albero).
            steps_needed = {g.game_idx: math.ceil(num_simulations / self.leaves_per_tree)
                            for g in active}

            while True:
                current = [g for g in active
                           if not g.done and steps_needed[g.game_idx] > 0]
                if not current:
                    break

                # Raccogli foglie da tutti gli alberi attivi
                # all_triples: (GameState, leaf_node, path_to_leaf)
                all_triples: list[tuple] = []
                for g in current:
                    leaf_path_pairs = self._select_leaves(g.root)
                    for leaf, path in leaf_path_pairs:
                        all_triples.append((g, leaf, path))

                # Deduplica per id del nodo (intra-tree: comune)
                seen_ids: dict[int, int] = {}
                unique_pairs: list[tuple[GameState, MCTSNode]] = []
                for g, leaf, path in all_triples:
                    nid = id(leaf)
                    if nid not in seen_ids:
                        seen_ids[nid] = len(unique_pairs)
                        unique_pairs.append((g, leaf))

                to_expand = [(g, n) for g, n in unique_pairs
                             if not n.is_terminal and not n.is_expanded]
                terminals = [(g, n) for g, n in unique_pairs
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
                # _backpropagate rimuove anche la virtual loss dal path
                for g, leaf, path in all_triples:
                    v = value_map.get(id(leaf), 0.0)
                    self._backpropagate(leaf, v, path)

                # Decrementa contatore per partita (uno step = un giro di foglie)
                for g in current:
                    steps_needed[g.game_idx] -= 1

            # ---- Scegli mossa per ogni partita attiva ----
            for g in active:
                if g.done or g.board.is_game_over() or g.move_num >= max_moves:
                    self._finalize_game(g)
                    continue

                temp = temp_high if g.move_num < temp_threshold else temp_low
                move = self._select_move(g.root, temp)

                # Salva step per il replay buffer (entrambi i lati)
                legal_moves_list = g.root.legal_moves
                legal_moves_t    = encode_legal_moves(g.board)
                policy_target    = self._visit_distribution(g.root, temp)

                if policy_target:  # salta se children vuoti
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
                        "value_target":  None,  # riempito a fine partita
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

        # ----------------------------------------------------------------
        # Assegna value_target (dal punto di vista di chi muove in ogni
        # step) e raccogli i risultati dal punto di vista del Bianco.
        # ----------------------------------------------------------------
        all_steps     = []
        white_results = []
        for g in games:
            white_result = self._white_result(g)
            for step in g.steps:
                board_turn = chess.Board(step["board_fen"]).turn
                step["value_target"] = white_result if board_turn == chess.WHITE else -white_result
            all_steps.append(g.steps)
            white_results.append(white_result)

        return all_steps, white_results

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
            leaf_path_pairs = self._select_leaves(root)
            unique = list({id(n): n for n, _ in leaf_path_pairs}.values())

            to_expand = [n for n in unique if not n.is_terminal and not n.is_expanded]
            terminals  = [n for n in unique if n.is_terminal or n.is_expanded]

            value_map: dict[int, float] = {}
            if to_expand:
                for n, v in zip(to_expand, self._expand_nodes_batched(to_expand)):
                    value_map[id(n)] = v
            for n in terminals:
                value_map[id(n)] = self._terminal_value(n)

            for leaf, path in leaf_path_pairs:
                self._backpropagate(leaf, value_map.get(id(leaf), 0.0), path)
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
    # Selezione foglie da un albero
    # ------------------------------------------------------------------

    def _select_leaves(self, root: MCTSNode) -> list[tuple]:
        """
        Seleziona leaves_per_tree foglie applicando virtual loss durante la discesa.
        Ritorna lista di (leaf, path) dove path include i nodi visitati (fog inclusa).
        La virtual loss diversifica le traiettorie successive nello stesso step.
        Viene rimossa durante il backprop reale in _backpropagate.
        """
        results = []
        for _ in range(self.leaves_per_tree):
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

    def _backpropagate(self, node: MCTSNode, value: float, path: list):
        """
        Backpropaga value risalendo l'albero da node fino alla radice.
        Rimuove la virtual loss dai nodi in path (applicata in _select_leaves).
        """
        # Rimuovi virtual loss prima di aggiornare i contatori reali
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
        if not moves:
            return {}
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
        return 1.0 if outcome.winner == node.board.turn else -1.0

    def _finalize_game(self, g: GameState):
        if not g.done:
            g.done   = True
            g.result = g.board.result()

    def _white_result(self, g: GameState) -> float:
        """Esito della partita dal punto di vista del Bianco: +1/-1/0."""
        result = g.result or g.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        return 0.0

    def _add_dirichlet_noise(self, root: MCTSNode):
        if not root.children:
            return
        children = list(root.children.values())
        noise    = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps      = self.dirichlet_epsilon
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * n