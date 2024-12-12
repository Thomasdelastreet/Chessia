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
def train_ai(model, optimizer, duration=600, save_interval=300, save_path="chess_ai.pth"):
    """
    Entraîne le modèle avec une durée limite. Sauvegarde régulièrement l'état du modèle.
    - `duration` : Durée totale de l'entraînement en secondes.
    - `save_interval` : Intervalle entre deux sauvegardes en secondes.
    - `save_path` : Chemin du fichier pour sauvegarder le modèle.
    """
    criterion = nn.MSELoss()
    start_time = time.time()
    iteration = 0

    while time.time() - start_time < duration:  # Boucle jusqu'à atteindre la durée limite
        board = chess.Board()
        inputs = torch.rand(1, 773)  # Exemple de représentation simplifiée
        labels = torch.tensor([[0.5]])  # Évaluation factice

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iteration += 1

        # Sauvegarde périodique
        if int(time.time() - start_time) % save_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"[{time.ctime()}] Modèle sauvegardé après {iteration} itérations. Perte : {loss.item():.4f}")

    # Sauvegarde finale avant arrêt
    torch.save(model.state_dict(), save_path)
    print(f"Entraînement terminé après {iteration} itérations. Modèle sauvegardé dans '{save_path}'.")

# --- Initialiser ou charger un modèle ---
model_path = "Chessia/chess_ai.pth"
model = ChessAI()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Modèle chargé depuis", model_path)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lancer l'entraînement avec limite de durée
train_ai(model, optimizer, duration=600, save_interval=300, save_path=model_path)
