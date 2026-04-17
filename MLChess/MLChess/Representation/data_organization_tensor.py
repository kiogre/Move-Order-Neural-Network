import chess
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, move_vocab, split='train', transform=None):
        """
        Inizializza il dataset leggendo il file CSV e dividendo i dati in train/validation/test.
        
        Args:
            csv_file (str): Percorso al file CSV contenente i dati.
            split (str): 'train', 'validation', o 'test'
            transform (callable, optional): Trasformazione da applicare ai dati.
        """
        # Leggi il file CSV
        self.df = pd.read_csv(csv_file)
        
        # Dividi i dati in train/validation/test (70% train, 15% validation, 15% test)
        total_len = len(self.df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        
        if split == 'train':
            self.df = self.df.iloc[:train_end]
        elif split == 'validation':
            self.df = self.df.iloc[train_end:val_end]
        elif split == 'test':
            self.df = self.df.iloc[val_end:]
        else:
            raise ValueError("split deve essere 'train', 'validation', o 'test'")
        
        self.df = self.df.reset_index(drop=True)
        
        # Estrai le colonne "FEN" e "Move"
        self.position = self.df["FEN"]
        self.move = self.df["Move"]
        self.result = self.df["Evaluation"]

        self.move_vocab = move_vocab

        # Imposta la trasformazione
        self.transform = transform

    def __len__(self):
        """
        Restituisce la lunghezza del dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Restituisce un singolo elemento del dataset alla posizione specificata dall'indice.
        
        Args:
            index (int): Indice dell'elemento da restituire.
        
        Returns:
            tuple: Una coppia (position, move), eventualmente trasformata.
        """
        row = self.df.iloc[index]
        position = row["FEN"]
        move = row["Move"]
        result = row["Evaluation"]
        board = chess.Board(position)
        legal_moves = [str(move) for move in board.legal_moves]
        mask = [self.move_vocab.get(m, -1) for m in legal_moves]

        if self.transform is not None:
            position, move, mask, result = self.transform(position, move, mask, result)

        return position, move, mask, result
    

class ChessTransform:
    def __init__(self, move_vocab):
        self.move_vocab = move_vocab
        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
        }

    def __call__(self, position, move, legal_indices, result):
        board_planes = torch.zeros((13, 8, 8), dtype=torch.float32)

        mask = [0] * 1968
        for idx in legal_indices:
            mask[idx] = 1


        board_fen = position.split(' ')[0]  # solo la parte dei pezzi
        turn = position.split(' ')[1]       # 'w' o 'b'

        rows = board_fen.split('/')
        for rank_idx, row in enumerate(rows):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                elif char in self.piece_to_plane:
                    plane_idx = self.piece_to_plane[char]
                    board_planes[plane_idx, rank_idx, file_idx] = 1
                    file_idx += 1

        # Piano 12 = turno
        board_planes[12, :, :] = 1 if turn == 'w' else 0

        # Converte la mossa
        move_encoded = self.move_vocab.get(move, -1)

        max_cp = 1000

        if '#' in result:
            result = 1 if '+' in result else -1
        else:
            result = max(-max_cp, min(max_cp, int(result)))  # clamp

        result /= max_cp

        return board_planes, move_encoded, mask, result


def collate_fn(batch):
    """
    Combina un batch di dati applicando padding alle sequenze FEN.
    
    Args:
        batch (list): Lista di tuple (position, move, length).
    
    Returns:
        positions_padded (Tensor): Sequenze FEN con padding.
        moves (Tensor): Tensor delle mosse.
        lengths (Tensor): Lunghezze originali delle sequenze FEN.
    """
    # Separa i dati dal batch
    positions, moves, mask, result = zip(*batch)
    
    # Converte le posizioni in tensori
    positions = [torch.Tensor(pos) for pos in positions]
    
    # Applica padding alle sequenze FEN
    positions_padded = pad_sequence(positions, batch_first=True, padding_value=0)
    
    # Converte le mosse in un tensore
    moves = torch.tensor(moves, dtype=torch.long)

    mask = torch.tensor(mask, dtype=bool)

    result = torch.tensor(result, dtype=torch.float)
    
    return positions_padded, moves, mask, result


def generate_all_legal_move_vocab() -> dict[str, int]:
    move_dict = {}
    index = 0

    promotion_pieces = ['q', 'r', 'b', 'n']

    for color in [chess.WHITE, chess.BLACK]:
        for from_square in chess.SQUARES:
            from_name = chess.square_name(from_square)
            file_from = chess.square_file(from_square)
            rank_from = chess.square_rank(from_square)

            for piece_type in [
                chess.PAWN,
                chess.KNIGHT,
                chess.QUEEN,
            ]:

                # ---- CAVALLO ----
                if piece_type == chess.KNIGHT:
                    deltas = [
                        (-1, -2), (1, -2),
                        (-2, -1), (2, -1),
                        (-2, 1), (2, 1),
                        (-1, 2), (1, 2)
                    ]
                    for df, dr in deltas:
                        f = file_from + df
                        r = rank_from + dr
                        if 0 <= f < 8 and 0 <= r < 8:
                            to_name = chess.square_name(chess.square(f, r))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index
                                index += 1

                # ---- REGINA ----
                elif piece_type == chess.QUEEN:
                    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for df, dr in directions:
                        f, r = file_from, rank_from
                        while True:
                            f += df
                            r += dr
                            if not (0 <= f < 8 and 0 <= r < 8):
                                break
                            to_name = chess.square_name(chess.square(f, r))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index
                                index += 1

                # ---- PEDONE ----
                elif piece_type == chess.PAWN:
                    if color == chess.WHITE:
                        # Avanti di uno
                        if rank_from < 7:
                            to_name = chess.square_name(chess.square(file_from, rank_from + 1))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index
                                index += 1
                            # Doppio passo iniziale
                            if rank_from == 1:
                                to_name = chess.square_name(chess.square(file_from, 3))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                            # Promozione
                            if rank_from == 6:
                                for promo in promotion_pieces:
                                    move_str = from_name + chess.square_name(chess.square(file_from, 7)) + promo
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index
                                        index += 1
                            # Catture diagonali
                            if file_from > 0:
                                to_name = chess.square_name(chess.square(file_from - 1, rank_from + 1))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                                if rank_from == 6:
                                    for promo in promotion_pieces:
                                        move_str = from_name + to_name + promo
                                        if move_str not in move_dict:
                                            move_dict[move_str] = index
                                            index += 1
                            if file_from < 7:
                                to_name = chess.square_name(chess.square(file_from + 1, rank_from + 1))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                                if rank_from == 6:
                                    for promo in promotion_pieces:
                                        move_str = from_name + to_name + promo
                                        if move_str not in move_dict:
                                            move_dict[move_str] = index
                                            index += 1
                    else:
                        # Pedone nero
                        if rank_from > 0:
                            to_name = chess.square_name(chess.square(file_from, rank_from - 1))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index
                                index += 1
                            if rank_from == 6:
                                to_name = chess.square_name(chess.square(file_from, 4))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                            if rank_from == 1:
                                for promo in promotion_pieces:
                                    move_str = from_name + chess.square_name(chess.square(file_from, 0)) + promo
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index
                                        index += 1
                            if file_from > 0:
                                to_name = chess.square_name(chess.square(file_from - 1, rank_from - 1))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                                if rank_from == 1:
                                    for promo in promotion_pieces:
                                        move_str = from_name + to_name + promo
                                        if move_str not in move_dict:
                                            move_dict[move_str] = index
                                            index += 1
                            if file_from < 7:
                                to_name = chess.square_name(chess.square(file_from + 1, rank_from - 1))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index
                                    index += 1
                                if rank_from == 1:
                                    for promo in promotion_pieces:
                                        move_str = from_name + to_name + promo
                                        if move_str not in move_dict:
                                            move_dict[move_str] = index
                                            index += 1

    return move_dict


def create_dataloaders_tensor(name_file: str = "over_mate_1_tactic_evals.csv", batch_size: int = 128, num_workers: int = 0, pin_memory: bool = False):
    '''
    Function to create dataloaders for training, validation and testing for chess, given a CSV file with this format:
    FEN,Move,Evaluation
    1. FEN: string representing the chess position in Forsyth-Edwards Notation.
    2. Move: string representing the best move in standard algebraic notation.
    3. Evaluation: integer representing the evaluation of the position (positive for white advantage, negative for black advantage, or special notation for checkmate).
    
    At the end I decided to return evene the result of generate_all_legal_move_vocab, dosen't have any sense to generate, destroy it and regenerate again.
    '''
    # Modifica della sezione di creazione dei dataset
    BATCH_SIZE = batch_size
    CSV_FILE = name_file

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()

    # Genera il vocabolario delle mosse
    all_moves = generate_all_legal_move_vocab()
    move_vocab = {move: idx for idx, move in enumerate(all_moves)}

    # Crea la trasformazione
    data_transforms = ChessTransform(move_vocab=move_vocab)

    # Crea dataset completi train/validation/test
    trainset = ChessDataset(csv_file=CSV_FILE, move_vocab=move_vocab, split='train', transform=data_transforms)
    validationset = ChessDataset(csv_file=CSV_FILE, move_vocab=move_vocab, split='validation', transform=data_transforms)
    testset = ChessDataset(csv_file=CSV_FILE, move_vocab=move_vocab, split='test', transform=data_transforms)

    # Caricamento dati
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        generator=g
    )

    validationloader = torch.utils.data.DataLoader(
        validationset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        generator=g
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        generator=g
    )

    print(f"Train set size: {len(trainset)}")
    print(f"Validation set size: {len(validationset)}")
    print(f"Test set size: {len(testset)}")

    #print(move_vocab)

    return trainloader, validationloader, testloader, move_vocab
