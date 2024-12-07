import chess
import chess.engine
import torch
import torch.nn as nn
import os

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

# --- Entraînement ---
def train_ai(model, optimizer, epochs=10):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        # Exemple de données factices pour l'entraînement (à adapter)
        board = chess.Board()
        inputs = torch.rand(1, 773)  # Représentation simplifiée
        labels = torch.tensor([[0.5]])  # Évaluation factice
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Entraînement terminé")
    return model

# Charger ou créer un modèle
model_path = "chess_ai.pth"
model = ChessAI()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

# Optimisation et entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train_ai(model, optimizer)

# Sauvegarder le modèle
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé dans {model_path}")
