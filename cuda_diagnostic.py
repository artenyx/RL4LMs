import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

count = torch.cuda.device_count()

for i in range(count):
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(i)/1024**3, 1), 'GB')