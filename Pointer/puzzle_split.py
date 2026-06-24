"""
puzzle_split.py — split deterministico train/probe per i puzzle Lichess.

Garantisce che PuzzleEvaluator (probe held-out, in eval_puzzles.py) e i
dataset usati per il training (CurriculumDataset2, MixedBufferDataset, in
train_alphazero_v3.py) operino su partizioni DISGIUNTE del CSV puzzle,
indipendentemente dai filtri sui Themes — quindi nessuna posizione del
probe può finire (con un value_target esplicito o come posizione di
partenza) nei dati di training, e viceversa.

Usa hashlib.md5 (non l'hash() built-in di Python, che e' salato in modo
casuale ad ogni processo per sicurezza) per ottenere uno split deterministico
e riproducibile sia tra esecuzioni diverse dello stesso script, sia tra
file diversi (eval_puzzles.py / train_alphazero_v3.py) che lo importano.
"""

import hashlib


def is_probe_puzzle(puzzle_id: str, probe_fraction: float = 0.1) -> bool:
    """
    True se questo puzzle appartiene alla partizione "probe" (held-out),
    False se appartiene alla partizione "train".

    probe_fraction: frazione approssimativa del dataset totale assegnata
    al probe (default 0.1 = 10%). Con dataset Lichess da centinaia di
    migliaia / milioni di puzzle, il 10% e' piu' che sufficiente per
    campionare i 300 puzzle di PuzzleEvaluator anche dopo il filtro sui
    temi mateIn1/2/3.

    Deterministico: stesso PuzzleId -> stesso risultato, sempre, in ogni
    processo/file che importa questa funzione.
    """
    h = int(hashlib.md5(str(puzzle_id).encode()).hexdigest(), 16)
    return (h % 1000) < int(probe_fraction * 1000)
