import chess
import chess.pgn
import torch
import torch.nn as nn
import random

# Modèle simple pour l'IA
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Convertir l'état de l'échiquier en entrée pour le réseau
def board_to_tensor(board):
    tensor = torch.zeros(64)
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece:
            tensor[i] = 1 if piece.color == chess.WHITE else -1
    return tensor

# Fonction d'entraînement
def train(model, games=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for game in range(games):
        board = chess.Board()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            best_move = random.choice(legal_moves)  # IA aléatoire basique
            board.push(best_move)
        
        # Calculer la récompense finale (1 pour victoire, 0 pour défaite)
        result = 1 if board.result() == '1-0' else 0
        state_tensor = board_to_tensor(board)
        output = model(state_tensor)
        loss = criterion(output, torch.tensor([result], dtype=torch.float))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Sauvegarder le modèle
model = ChessNet()
train(model)
torch.save(model.state_dict(), "chess_ai.pth")
