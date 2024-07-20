import torch

file_path = '/data1/akamaharuka/data//train_X.pt'
try:
    data = torch.load(file_path)
    print("File loaded successfully.")
except Exception as e:
    print(f"Failed to load the file: {e}")
