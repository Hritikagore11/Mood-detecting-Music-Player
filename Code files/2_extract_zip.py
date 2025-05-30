
import zipfile
import os
import shutil

zip_path = '/content/drive/MyDrive/music/archive (1).zip'
extract_path = '/content/music'

# Clear existing folders if they exist
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Folders inside /content/music:", os.listdir(extract_path))
