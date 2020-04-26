from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io
import os
import pickle
import sys

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=1337)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token, protocol=0)

    service = build('drive', 'v3', credentials=creds)

    file_name = ''
    location = ''
    file_id = ''
    if len(sys.argv) > 2:
        location = sys.argv[2]
        if location[-1] != '/':
            location += '/'

    _file = service.files().list(
            q=f"name contains {sys.argv[1]}",
            fields='files(id, name)').execute()

    total = len(_file['files'])
    if total != 1:
        print(f'{total} folders found')
        if total == 0:
            sys.exit(1)
        prompt = 'Please select the folder you want to download:\n\n'
        for i in range(total):
            prompt += f'[{i}]: {get_full_path(service, _file["files"][i])}\n'
        prompt += '\nYour choice: '
        choice = int(input(prompt))
        if 0 <= choice and choice < total:
            file_id = _file['files'][choice]['id']
            file_name = _file['files'][choice]['name']
        else:
            sys.exit(1)
    else:
        file_id = _file['files'][0]['id']
        file_name = _file['files'][0]['name']

    print(f'{file_id} {file_name}')
    download_file(service, file_id, location, file_name)
    
def download_file(service, file_id, location, file_name):

    print(location)
    print(file_name)

    if not os.path.exists(location):
        os.makedirs(location)

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(location + file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        try:
            status, done = downloader.next_chunk()
        except:
            fh.close()
            os.remove(location + file_name)
            sys.exit(1)
        print(f'\rDownload {int(status.progress() * 100)}%.', end='')
        sys.stdout.flush()
    print('')

if __name__ == '__main__':
    main()
