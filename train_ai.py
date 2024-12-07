import chess
import chess.engine
import json
import time

# Chemin du moteur d'échecs (exemple : Stockfish)
ENGINE_PATH = "stockfish"
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

# Fonction d'entraînement
def train_ai(training_file="training_data.json"):
    training_data = []

    try:
        # Charger les données existantes
        with open(training_file, "r") as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print("Fichier d'entraînement non trouvé. Création d'un nouveau fichier.")

    while True:
        board = chess.Board()
        moves = []
        while not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            moves.append(result.move.uci())

        training_data.append({"moves": moves, "result": board.result()})

        # Sauvegarder après chaque partie
        with open(training_file, "w") as f:
            json.dump(training_data, f)

        print("Partie terminée, sauvegardée dans", training_file)
        time.sleep(1)  # Éviter une surcharge des ressources

if __name__ == "__main__":
    train_ai()
