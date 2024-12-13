from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
import os

# Chemin des credentials JSON (fourni via les secrets GitHub)
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Initialiser les credentials
SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS, scopes=SCOPES)

# Construire le service Google Drive
drive_service = build('drive', 'v3', credentials=creds)

# Définir le fichier à uploader
FILE_NAME = "chess_ai.pth"
file_metadata = {'name': FILE_NAME}
media = MediaFileUpload(FILE_NAME, mimetype='application/octet-stream')

# Uploader sur Google Drive
file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print(f"File uploaded successfully! File ID: {file.get('id')}")
