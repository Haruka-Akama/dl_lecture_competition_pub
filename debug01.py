import torch
import GPUtil
from tabulate import tabulate

def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    gpu_list = []
    for gpu in gpus:
        gpu_list.append((
            gpu.id, gpu.name, f"{gpu.load*100:.0f}%", 
            f"{gpu.memoryUsed}MB", f"{gpu.memoryTotal}MB", 
            f"{gpu.memoryUtil*100:.0f}%"))
    
    print(tabulate(gpu_list, headers=("ID", "Name", "Load", "Memory Used", "Total Memory", "Memory Utilization")))

if __name__ == "__main__":
    print_gpu_utilization()
