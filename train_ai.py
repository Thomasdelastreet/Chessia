import chess
import torch
import torch.nn as nn
import os
import time

# --- Définir le modèle IA ---
class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(773, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)

# --- Fonction d'entraînement ---
def train_ai(model, optimizer, save_interval=100, save_path="chess_ai.pth"):
    criterion = nn.MSELoss()
    iteration = 0

    while True:  # Boucle infinie
        board = chess.Board()
        inputs = torch.rand(1, 773)  # Exemple de représentation simplifiée
        labels = torch.tensor([[0.5]])  # Évaluation factice

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iteration += 1

        if iteration % save_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"[{time.ctime()}] Modèle sauvegardé après {iteration} itérations. Perte : {loss.item():.4f}")

# --- Initialiser ou charger un modèle ---
model_path = "chess_ai.pth"
model = ChessAI()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lancer l'entraînement en boucle infinie
train_ai(model, optimizer)
