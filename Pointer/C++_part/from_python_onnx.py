import torch
import torch.nn as nn
from MLChess import JellyFishPointer

model = JellyFishPointer()
model.eval()

# Dummy input con 20 mosse di test
dummy_board = torch.randn(1, 13, 8, 8, dtype=torch.float32)
dummy_moves = torch.randn(1, 20, 46, dtype=torch.float32)
dummy_mask = torch.ones(1, 20, dtype=torch.bool)

torch.onnx.export(
    model,
    (dummy_board, dummy_moves, dummy_mask),
    "jellyfish_pointer.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['board', 'moves', 'move_mask'],
    output_names=['logits', 'probs', 'value'],
    dynamic_axes={
        'board': {0: 'batch_size'},
        'moves': {0: 'batch_size', 1: 'n_mosse'},
        'move_mask': {0: 'batch_size', 1: 'n_mosse'},
        'logits': {0: 'batch_size', 1: 'n_mosse'},
        'probs': {0: 'batch_size', 1: 'n_mosse'},
        'value': {0: 'batch_size'}
    }
)