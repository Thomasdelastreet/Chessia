from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

# Initialiser les credentials
SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)

# Construire le service Drive
drive_service = build('drive', 'v3', credentials=creds)

# Définir le fichier à uploader
file_metadata = {'name': 'chess_ai.pth'}
media = MediaFileUpload('chess_ai.pth', mimetype='application/octet-stream')

# Uploader sur Google Drive
file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print(f"File uploaded successfully! File ID: {file.get('id')}")
