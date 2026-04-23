import torch

def rook_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    for i in range(64):
        for j in range(64):
            if i == j:
                moves[i, j] = 1.0
            elif (i // 8 == j // 8 or i % 8 == j % 8):
                moves[i, j] = 0.5 

    moves[moves == 0] = 1/3
    return moves

def bishop_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    for i in range(64):
        r1, c1 = i // 8, i % 8
        color1 = (r1 + c1) % 2
        
        for j in range(64):
            r2, c2 = j // 8, j % 8
            color2 = (r2 + c2) % 2
            
            if i == j:
                moves[i, j] = 1.0
            elif color1 != color2:
                # Colore diverso: l'alfiere non può mai arrivarci
                moves[i, j] = 0.0
            elif abs(r1 - r2) == abs(c1 - c2):
                moves[i, j] = 0.5
            else:
                moves[i, j] = 1/3
                
    return moves

def queen_moves():
    rook = rook_moves()
    bishop = bishop_moves()
    return torch.max(rook, bishop)

def king_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    for i in range(64):
        r1, c1 = i // 8, i % 8
        
        for j in range(64):
            r2, c2 = j // 8, j % 8

            # Scegliere il massimo, poi fare distanza di Manhattan            
            moves[i,j] = 1/(1 + abs(r1 - r2)*(abs(r1-r2)>=abs(c1-c2)) + abs(c1 - c2)*(abs(c1-c2)>abs(r1-r2)))
            
                
    return moves

def white_pawn_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    for i in range(8, 64):
        r1, c1 = i // 8, i % 8
        
        for j in range(64):
            r2, c2 = j // 8, j % 8
            
            if i == j:
                moves[i, j] = 1.0
            elif c2 == c1 and r2>r1:
                moves[i,j] = 1/(1 + (r2 - r1)-int(r1==1))

    for i in range(8):
        moves[i+8, 16+i] = 0.5
    return moves

def black_pawn_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    for i in range(56):
        r1, c1 = i // 8, i % 8
        
        for j in range(64):
            r2, c2 = j // 8, j % 8
            
            if i == j:
                moves[i, j] = 1.0
            elif c2 == c1 and r2<r1:
                moves[i,j] = 1/(1 + (r1 - r2)-int(r1==6))

    for i in range(8):
        moves[i+48, 40+i] = 0.5
    return moves

import torch
from collections import deque

def knight_moves():
    moves = torch.zeros((64, 64), dtype=torch.float32)
    
    offsets = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    
    for start_node in range(64):
        distances = [-1] * 64
        distances[start_node] = 0
        queue = deque([start_node])
        
        while queue:
            curr = queue.popleft()
            r, c = curr // 8, curr % 8
            
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    neighbor = nr * 8 + nc
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[curr] + 1
                        queue.append(neighbor)

        for target_node in range(64):
            n = distances[target_node]
            moves[start_node, target_node] = 1.0 / (n + 1)
            
    return moves
'''
import matplotlib.pyplot as plt

moves = rook_moves()

e2 = moves[0]

e2 = moves.mean(dim=0).numpy().reshape(64)

for i in range(64):
    print(f"{float(e2[i]):.10f}" , end=" ")
    if (i+1)%8==0:
        print('')
'''

all_moves = {
    'rook': rook_moves(),
    'bishop': bishop_moves(),
    'queen': queen_moves(),
    'king': king_moves(),
    'white_pawn': white_pawn_moves(),
    'black_pawn': black_pawn_moves(),
    'knight': knight_moves()
}

M = torch.stack([
    all_moves['white_pawn'],
    all_moves['black_pawn'],
    all_moves['knight'],
    all_moves['bishop'],
    all_moves['rook'],
    all_moves['queen'],
    all_moves['king'],
])  # (7, 64, 64)

torch.save(M, 'attention_based_matrix_64x64.pt')