import zipfile

zip_file_path = '/data1/akamaharuka/data/train_X.pt'
extract_path = '/data1/akamaharuka/data/extracted/'

# ディレクトリが存在しない場合は作成
import os
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

# ZIPファイルを解凍
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"ZIPファイル {zip_file_path} を {extract_path} に解凍しました。")
except zipfile.BadZipFile:
    print(f"{zip_file_path} はZIPファイルではありません。")
except Exception as e:
    print(f"ZIPファイルの解凍中にエラーが発生しました: {e}")
