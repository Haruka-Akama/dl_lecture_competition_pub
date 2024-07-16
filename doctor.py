import os

def list_files(directory):
    print(f"Listing files in directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

# ディレクトリのパスを指定
directory = "/data1/akamaharuka/data-omni/"

# ディレクトリ内のすべてのファイル名を出力
list_files(directory)
