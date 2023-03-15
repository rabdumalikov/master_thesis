from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    mallocs = []#[ torch.cuda.memory_allocated(f'cuda:{c}')*0.000001 for c in range(2) ]
    
    print(f"\t GPU memory occupied: {info.used//1024**2} MB. {mallocs=} MB")
