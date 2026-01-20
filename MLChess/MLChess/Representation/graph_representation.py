from torch_geometric.data import Data, InMemoryDataset, Dataset
import chess
import torch
import gc
from typing import Iterator, List, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import h5py
from multiprocessing import Pool, cpu_count

## HAVE TO UPDATE NEW IMPLEMENTATION GRAPH WITH THE MOVE
## HAVE TO UPDATE IMPLEMENTATION TENSOR WITH DIFFERENT KIND MOVE, BUT THIS MUCH LATER

BB_ROOK_ATTACKS: List[chess.Bitboard] = [chess._step_attacks(sq, [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, -1, -2, -3, -4, -5, -6, -7, -8, -16, -24, -32, -40, -48, -56]) for sq in chess.SQUARES]
BB_BISHOP_ATTACKS: List[chess.Bitboard] = [chess._step_attacks(sq, [9, 18, 27, 36, 45, 54, 63, 7, 14, 21, 28, 35, 42, 49, 56, -9, -18, -27, -36, -45, -54, -63, -7, -14, -21, -28, -35, -42, -49, -56]) for sq in chess.SQUARES]

def precompute_queen_knight_edges():
    edges = []

    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        # ---- KNIGHT ----
        for d_rank, d_file in [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]:
            r, f = rank + d_rank, file + d_file
            if 0 <= r < 8 and 0 <= f < 8:
                edges.append([sq, chess.square(f, r)])

        # ---- QUEEN ----
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for d_rank, d_file in directions:
            r, f = rank + d_rank, file + d_file
            while 0 <= r < 8 and 0 <= f < 8:
                edges.append([sq, chess.square(f, r)])
                r += d_rank
                f += d_file

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def move_to_edge(move_uci: str):
    from_sq = chess.parse_square(move_uci[:2])
    to_sq = chess.parse_square(move_uci[2:4])
    return from_sq, to_sq


class ChessPositionGraph:
    """
    Most importante thing: this is just 64 nodes, but the arch can be modified: or only possible moves
    or all the moves, just the parameter minimal arch
    """

    def __init__(self, legal_move_graph: bool = False):
        self.piece_type_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        self.piece_value_map = {
            'P': 0.1,  'N': 0.325, 'B': 0.3, 'R': 0.5, 'Q': 0.9, 'K': 1.0,
            'p': -0.1, 'n': -0.325,'b': -0.3,'r': -0.5,'q': -0.9,'k': -1.0
        }
        self.edge_index = precompute_queen_knight_edges()
        self.minimal = legal_move_graph
        self.edge_to_index = {
            (src, dst): i
            for i, (src, dst) in enumerate(self.edge_index.t().tolist())
        }


    
    def fen_to_graph(self, fen_string: str, evaluation: str, best_move, max_evaluation: int = 1000) -> Data:
        """Converte FEN in grafo"""
        
        # 1. Gestione valutazione
        if '#' in evaluation:
            eval_value = 1.0 if '+' in evaluation else -1.0
        else:
            try:
                evaluation = evaluation.replace('+', '')
                eval_value = int(evaluation)
                eval_value = eval_value / max_evaluation
                eval_value = max(-1.0, min(1.0, eval_value))
            except (ValueError, TypeError):
                eval_value = 0.0
        
        # 2. Features nodi
        board = chess.Board(fen_string)

        num_edges = self.edge_index.size(1)
        legal_edge_mask = torch.zeros(num_edges, dtype=torch.bool)

        for move in board.legal_moves:
            src = move.from_square
            dst = move.to_square

            # ignoriamo promozioni per ora, poi le estendiamo
            idx = self.edge_to_index.get((src, dst))
            if idx is not None:
                legal_edge_mask[idx] = 1


        node_features = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_features = [0] * 12
                piece_features[self.piece_type_map[piece.symbol()]] = 1
                color_feature = 1.0 if piece.color == chess.WHITE else -1.0
                node_features.append(piece_features + [(square % 8)/7.0] + [(square // 8)/7.0] + [self.piece_value_map[piece.symbol()]])
            else:
                node_features.append([0] * 12 + [(square % 8)/7.0] + [(square // 8)/7.0] + [0])

        # 3. Gestione degli archi
        if self.minimal:
            edge_index = []
            for move in board.legal_moves:
                edge_index.append([move.from_square, move.to_square])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = self.edge_index

        policy_target = torch.zeros(self.edge_index.size(1), dtype=torch.float)

        if best_move is not None:
            try:
                src, dst = move_to_edge(best_move)
                edge_idx = self.edge_to_index.get((src, dst), None)
                if edge_idx is not None:
                    policy_target[edge_idx] = 1.0
            except:
                pass

        
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([eval_value], dtype=torch.float)

        global_features = []
        
        # Turno (chi deve muovere)
        turn = 1.0 if board.turn == chess.WHITE else -1.0
        global_features.append(turn)
        
        # Diritti di arrocco
        global_features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
        global_features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
        global_features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
        global_features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        
        # En passant
        global_features.append(1.0 if board.ep_square is not None else 0.0)
        
        # Numero mosse
        global_features.append(board.fullmove_number / 100.0)

        global_features_tensor = torch.tensor(global_features, dtype=torch.float)
        
        return Data(
            x=x,
            edge_index=self.edge_index,      # SEMPRE quello completo
            global_features=global_features_tensor,
            y=y,
            y_policy=policy_target,
            legal_edge_mask=legal_edge_mask
        )

    

class CSVChessDataset:
    """Dataset che legge CSV a chunks con supporto per divisione train/test
    The representation of the chessboard can be implemented in different ways, sot can be done by 
    changing parameter representation
    """
    
    def __init__(self, csv_file: str, batch_size: int = 64, max_evaluation: int = 1000, 
                 indices: Set[int] = None, representation = ChessPositionGraph()):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.max_evaluation = max_evaluation
        self.indices = indices  # Set degli indici da usare (None = tutti)
        self.graph_converter = representation
        
        # Conta righe
        with open(csv_file, 'r') as f:
            self.total_positions = sum(1 for _ in f) - 1  # -1 per header
        
        if self.indices is not None:
            self.total_positions = len(self.indices)
            print(f"Dataset subset: {self.total_positions} posizioni")
        else:
            print(f"Dataset completo: {self.total_positions} posizioni")
    
    def get_batch_iterator(self, shuffle: bool = True) -> Iterator[List]:
        """Legge CSV a chunks e restituisce batch, filtrando per indici se specificato"""
        chunk_size = max(self.batch_size * 20, 100000)
        row_index = 0
        
        for chunk_df in pd.read_csv(self.csv_file, chunksize=chunk_size):
            chunk_graphs = []
            
            for local_idx, row in enumerate(chunk_df.itertuples(index=False)):
                current_row_index = row_index + local_idx
                
                # Se abbiamo indici specifici, controlla se questo indice è incluso
                if self.indices is not None and current_row_index not in self.indices:
                    continue
                
                try:
                    fen_string = row.FEN
                    evaluation = row.Evaluation
                    best_move = row.Move
                    graph_data = self.graph_converter.fen_to_graph(
                        fen_string, evaluation, best_move, self.max_evaluation
                    )
                    chunk_graphs.append(graph_data)
                except Exception as e:
                    continue
            
            row_index += len(chunk_df)
            
            if shuffle:
                np.random.shuffle(chunk_graphs)
            
            # Restituisci batch
            for i in range(0, len(chunk_graphs), self.batch_size):
                batch = chunk_graphs[i:i + self.batch_size]
                if len(batch) > 0:
                    yield batch
            
            del chunk_graphs, chunk_df
            gc.collect()


def create_train_test_split(csv_file: str, test_size: float = 0.2, val_size: float = 0.1):
    """
    Crea gli indici per train/val/test split senza caricare tutto il dataset
    """
    # Conta le righe totali
    with open(csv_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # -1 per header
    
    # Crea tutti gli indici
    all_indices = list(range(total_rows))
    
    # Prima divisione: train+val vs test
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Seconda divisione: train vs val
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size/(1-test_size), random_state=42, shuffle=True
    )

    return set(train_indices), set(val_indices), set(test_indices)


def dataset_creation_graph(csv_file: str = "over_mate_1_tactic_evals.csv", val_size: float = 0.15, test_size: float = 0.15, 
                           batch_size: int = 128, representation = ChessPositionGraph()):
    """
    Returns 3 objects that represents the dataset, training, validation and test.
    This function technically should make datasets for the graph representation of the chessboard.
    I'm not sure about how good it is, I should check the code again to be sure it isn't something really bad.
    Or if I can use some other thing.
    It doesn't work exactly like a dataloader.
    If I want to check again how I worked with it, I should go to the folder ChessNN and look at the .ipynb where I work with GNN.
    
    Can be set different kinds of representations, just modifiyng the parameter representation into a new class
    IMPORTANT: THE NEW IMPLEMENTATION SHOULD HAVE A FUNCTION CALLED fen_to_graph() THAT TAKES A STRING (FEN),
    VALUE OF THE POSITION AND A MAXIMUM VALUE THAT THE POSITION CAN ASSUME (DIFFERENT FROM 0)
    THE OUTPUT OF THIS FUNCTION IS: Data(
            x=x,        # [64, 13] - una riga per casella
            edge_index=edge_index,
            global_features=global_features_tensor,
            y=y
        )
    WHERE DATA IS FROM torch_geometric
    """
    train_indices, val_indices, test_indices = create_train_test_split(
        csv_file, test_size, val_size
    )

    print(f"Dataset diviso in:")
    print(f"  Training: {len(train_indices)} campioni ({len(train_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    print(f"  Validation: {len(val_indices)} campioni ({len(val_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    print(f"  Test: {len(test_indices)} campioni ({len(test_indices)/(len(train_indices)+len(val_indices)+len(test_indices))*100:.1f}%)")
    

    train_dataset = CSVChessDataset(csv_file, batch_size=batch_size, indices=train_indices, representation=representation)
    val_dataset = CSVChessDataset(csv_file, batch_size=batch_size, indices=val_indices, representation=representation)
    test_dataset = CSVChessDataset(csv_file, batch_size=batch_size, indices=test_indices, representation=representation)

    return train_dataset, val_dataset, test_dataset

def preprocess_to_hdf5(csv_file: str, output_file: str):
    df = pd.read_csv(csv_file)
    graph_converter = ChessPositionGraph()
    
    n_samples = len(df)
    n_nodes = 64
    n_node_features = 15
    n_edges = graph_converter.edge_index.size(1)
    
    with h5py.File(output_file, 'w') as f:
        # Pre-alloca con compressione
        x_dset = f.create_dataset('x', shape=(n_samples, n_nodes, n_node_features), 
                                   dtype='float32', compression='gzip', compression_opts=4)
        y_dset = f.create_dataset('y', shape=(n_samples, 1), dtype='float32')
        gf_dset = f.create_dataset('global_features', shape=(n_samples, 7), dtype='float32')
        yp_dset = f.create_dataset('y_policy', shape=(n_samples, n_edges), dtype='float32')
        mask_dset = f.create_dataset('legal_edge_mask', shape=(n_samples, n_edges), dtype='bool')
        
        f.create_dataset('edge_index', data=graph_converter.edge_index.numpy())
        
        # Processa tutto
        for i in tqdm(range(n_samples), desc="Preprocessing"):
            row = df.iloc[i]
            graph = graph_converter.fen_to_graph(row['FEN'], row['Evaluation'], row['Move'])
            
            x_dset[i] = graph.x.numpy()
            y_dset[i] = graph.y.numpy()
            gf_dset[i] = graph.global_features.numpy()
            yp_dset[i] = graph.y_policy.numpy()
            mask_dset[i] = graph.legal_edge_mask.numpy()
            
            if i % 10000 == 0:
                gc.collect()
    
    print(f"✓ Done! File size: {os.path.getsize(output_file) / 1e9:.2f} GB")


def process_batch(args):
    """
    Funzione worker per processare un batch di posizioni.
    Restituisce risultati PICCOLI per non riempire la RAM.
    """
    batch_data, graph_converter_params = args
    
    from MLChess.Representation.graph_representation import ChessPositionGraph
    graph_converter = ChessPositionGraph(**graph_converter_params)
    
    results = []
    for idx, row in batch_data.iterrows():
        try:
            graph = graph_converter.fen_to_graph(
                row['FEN'],  # Aggiusta con il nome corretto
                row['Evaluation'],
                row['Move']
            )
            
            results.append({
                'x': graph.x.numpy(),
                'y': graph.y.numpy(),
                'global_features': graph.global_features.numpy(),
                'y_policy': graph.y_policy.numpy(),
                'legal_edge_mask': graph.legal_edge_mask.numpy()
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append(None)
    
    return results


def preprocess_to_hdf5_fast(csv_file: str, output_file: str, 
                                     batch_size: int = 500,
                                     n_workers: int = None,
                                     max_queue_size: int = 4):
    """
    Versione MEMORY-SAFE con multiprocessing.
    Processa batch in parallelo MA scrive immediatamente su disco
    mantenendo solo pochi batch in RAM contemporaneamente.
    
    max_queue_size: numero MAX di batch in RAM (più alto = più veloce ma più RAM)
    """
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)
    
    print(f"🚀 Using {n_workers} workers (Memory-Safe Mode)")
    print(f"📦 Batch size: {batch_size}, Max queue: {max_queue_size} batches")
    
    # Leggi CSV
    print("📖 Reading CSV...")
    df = pd.read_csv(csv_file)
    n_samples = len(df)
    print(f"✓ Loaded {n_samples} positions")
    
    # Parametri
    from MLChess.Representation.graph_representation import ChessPositionGraph
    graph_converter = ChessPositionGraph()
    
    n_nodes = 64
    n_node_features = 15
    n_edges = graph_converter.edge_index.size(1)
    
    # Crea file HDF5
    print("📝 Creating HDF5 file...")
    with h5py.File(output_file, 'w') as f:
        # Senza compressione per velocità
        x_dset = f.create_dataset('x', shape=(n_samples, n_nodes, n_node_features), dtype='float32')
        y_dset = f.create_dataset('y', shape=(n_samples, 1), dtype='float32')
        gf_dset = f.create_dataset('global_features', shape=(n_samples, 7), dtype='float32')
        yp_dset = f.create_dataset('y_policy', shape=(n_samples, n_edges), dtype='float32')
        mask_dset = f.create_dataset('legal_edge_mask', shape=(n_samples, n_edges), dtype='bool')
        f.create_dataset('edge_index', data=graph_converter.edge_index.numpy())
        
        # Prepara i batch
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()  # .copy() importante!
            batches.append((batch_df, {}))
        
        print(f"⚙️  Processing {len(batches)} batches...")
        
        # CHIAVE: usa imap con chunksize=1 per processare in streaming
        with Pool(processes=n_workers) as pool:
            idx = 0
            
            # imap restituisce risultati nell'ordine originale
            # maxtasksperchild libera memoria dopo ogni N task
            results_iter = pool.imap(process_batch, batches, chunksize=1)
            
            for batch_results in tqdm(results_iter, total=len(batches), desc="Preprocessing"):
                # Scrivi immediatamente questo batch
                for result in batch_results:
                    if result is not None:
                        x_dset[idx] = result['x']
                        y_dset[idx] = result['y']
                        gf_dset[idx] = result['global_features']
                        yp_dset[idx] = result['y_policy']
                        mask_dset[idx] = result['legal_edge_mask']
                        idx += 1
                
                # LIBERA MEMORIA immediatamente
                del batch_results
                
                # Flush ogni 10 batch
                if idx % (batch_size * 10) == 0:
                    f.flush()
                    gc.collect()
            
            # Flush finale
            f.flush()
    
    file_size = os.path.getsize(output_file) / 1e9
    print(f"✅ Done! Processed {idx} positions")
    print(f"📦 File size: {file_size:.2f} GB")
    print(f"💾 Saved to: {output_file}")


class ChessPositionGraphMPNN:
    """
    Converte FEN in grafo pronto per MPNN
    """

    def __init__(self, legal_move_graph: bool = False):
        self.piece_type_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        self.piece_value_map = {
            'P': 0.1, 'N': 0.325, 'B': 0.3, 'R': 0.5, 'Q': 0.9, 'K': 1.0,
            'p': -0.1, 'n': -0.325,'b': -0.3,'r': -0.5,'q': -0.9,'k': -1.0
        }
        self.minimal = legal_move_graph
        self.edge_index = precompute_queen_knight_edges()  # come prima
        self.edge_to_index = {
            (src, dst): i
            for i, (src, dst) in enumerate(self.edge_index.t().tolist())
        }

    def fen_to_graph(self, fen_string: str, evaluation: str = "0", best_move=None, max_evaluation: int = 1000) -> Data:
        board = chess.Board(fen_string)

        # --- 1. Node features ---
        node_features = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_onehot = [0]*12
                piece_onehot[self.piece_type_map[piece.symbol()]] = 1
                node_features.append(piece_onehot + [(square % 8)/7.0, (square // 8)/7.0, self.piece_value_map[piece.symbol()]])
            else:
                node_features.append([0]*12 + [(square % 8)/7.0, (square // 8)/7.0, 0.0])

        x = torch.tensor(node_features, dtype=torch.float)

        # --- 2. Edge index ---
        if self.minimal:
            edges = [[m.from_square, m.to_square] for m in board.legal_moves]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = self.edge_index

        # --- 3. Edge features ---
        edge_attr_list = []
        for src, dst in edge_index.t().tolist():
            piece_src = board.piece_at(src)
            piece_dst = board.piece_at(dst)
            type_src = self.piece_type_map[piece_src.symbol()] if piece_src else 12
            type_dst = self.piece_type_map[piece_dst.symbol()] if piece_dst else 12
            delta_rank = (dst // 8 - src // 8)/7.0
            delta_file = (dst % 8 - src % 8)/7.0
            edge_attr_list.append([type_src, type_dst, delta_rank, delta_file])
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        # --- 4. Global features ---
        global_features = torch.tensor([
            1.0 if board.turn == chess.WHITE else -1.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.ep_square is not None else 0.0,
            board.fullmove_number / 100.0
        ], dtype=torch.float)

        # --- 5. Target ---
        y = torch.tensor([self._normalize_eval(evaluation, max_evaluation)], dtype=torch.float)

        # --- 6. Legal mask per policy ---
        legal_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for move in board.legal_moves:
            idx = self.edge_to_index.get((move.from_square, move.to_square))
            if idx is not None:
                legal_edge_mask[idx] = 1

        policy_target = torch.zeros(edge_index.size(1), dtype=torch.float)
        if best_move:
            try:
                src, dst = move_to_edge(best_move)
                edge_idx = self.edge_to_index.get((src, dst), None)
                if edge_idx is not None:
                    policy_target[edge_idx] = 1.0
            except:
                pass

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            y=y,
            y_policy=policy_target,
            legal_edge_mask=legal_edge_mask
        )

    def _normalize_eval(self, evaluation, max_eval=1000):
        try:
            evaluation = evaluation.replace('+', '')
            val = int(evaluation)/max_eval
            return max(-1.0, min(1.0, val))
        except:
            return 0.0