from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

# Chemin du fichier client_secrets.json
GOOGLE_CREDENTIALS = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Charger les credentials depuis le fichier JSON
creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS, scopes=SCOPES)

# Initialiser le service Google Drive
service = build('drive', 'v3', credentials=creds)

# Télécharger le fichier sur Google Drive
file_metadata = {'name': 'chess_ai.pth'}
media = MediaFileUpload('chess_ai.pth', mimetype='application/octet-stream')

# Upload
file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print(f"Fichier uploadé avec succès : {file.get('id')}")
