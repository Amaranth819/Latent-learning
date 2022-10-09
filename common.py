import torch
import numpy as np

'''
    Cast a np array to tensor
'''
def np_to_tensor(np_array, device = torch.device('cpu')):
    return torch.from_numpy(np_array).float().to(device)


'''
    Cast a tensor to np array
'''
def tensor_to_np(tensor : torch.Tensor):
    return tensor.detach().cpu().numpy()


'''
    Get device from string
'''
def get_device(s):
    assert s in ['auto', 'cpu', 'cuda']
    if s == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(s)