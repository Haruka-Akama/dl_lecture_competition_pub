import torch
import os

data_dir = "/data1/akamaharuka/data/"
file_path = os.path.join(data_dir, "train_X.pt")

def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = torch.load(f)
            print("Data loaded successfully")
            return data
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    load_data(file_path)
