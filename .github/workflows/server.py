import threading
import time
import chess
import chess.pgn
import torch
import torch.nn as nn
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

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

# Initialisation du modèle
model = ChessNet()

# Fonction d'entraînement
def train_ai():
    global model
    while True:
        board = chess.Board()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            best_move = random.choice(legal_moves)  # IA aléatoire basique
            board.push(best_move)
        
        time.sleep(0.01)  # Contrôle de la vitesse (100 parties/seconde)

# Thread pour l'entraînement continu
def background_training():
    threading.Thread(target=train_ai, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

# Gérer les mouvements du joueur
@socketio.on("player_move")
def handle_player_move(data):
    # Logique pour traiter les mouvements
    move = data["move"]
    emit("ai_move", {"move": move})  # Réponse fictive de l'IA

if __name__ == "__main__":
    background_training()
    socketio.run(app, debug=True)
