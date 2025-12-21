from torch_geometric.data import Data
import chess
import torch
import gc
from typing import Iterator, List, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ChessPositionGraph:
    """Converte FEN in grafo usando archi fissi pre-calcolati"""
    
    def __init__(self):
        self.piece_type_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
    
    def fen_to_graph(self, fen_string: str, evaluation: str, max_evaluation: int = 1000) -> Data:
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
        node_features = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_features = [0] * 12
                piece_features[self.piece_type_map[piece.symbol()]] = 1
                color_feature = 1.0 if piece.color == chess.WHITE else -1.0
                node_features.append(piece_features + [color_feature])
            else:
                node_features.append([0] * 12 + [0.0])

        # 3. Gestione degli archi
        edge_index = []
        for move in board.legal_moves:
            edge_index.append([move.from_square, move.to_square])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
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
            x=x,        # [64, 13] - una riga per casella
            edge_index=edge_index,
            global_features=global_features_tensor,
            y=y
        )
    

class CSVChessDataset:
    """Dataset che legge CSV a chunks con supporto per divisione train/test"""
    
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
        chunk_size = max(self.batch_size * 20, 10000)
        row_index = 0
        
        for chunk_df in pd.read_csv(self.csv_file, chunksize=chunk_size):
            chunk_graphs = []
            
            for local_idx, (_, row) in enumerate(chunk_df.iterrows()):
                current_row_index = row_index + local_idx
                
                # Se abbiamo indici specifici, controlla se questo indice è incluso
                if self.indices is not None and current_row_index not in self.indices:
                    continue
                
                try:
                    fen_string = row['FEN']
                    evaluation = str(row['Evaluation'])
                    graph_data = self.graph_converter.fen_to_graph(
                        fen_string, evaluation, self.max_evaluation
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
    If I want to check again how I worked with it, I should go to the folder ChessNN and look at the .ipynb where I work with GNN
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
