from collections import OrderedDict
import torch
import copy
# import cvxopt
# from cvxopt import matrix, solvers
import numpy as np
import torch.nn as nn


def evaluation(model, dataloader, criterion, model_params=None, device=None, eval_full_data=True):
    if model_params is not None:
        model.load_state_dict(model_params)

    if device is not None:
        model.to(device)

    model.eval()
    loss = 0.0
    acc = 0.0
    num = 0
    
    i = 0
    for x, y in dataloader:
        torch.cuda.empty_cache()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        with torch.no_grad():
            logit = model(x) ##
            _loss = criterion(logit, y)
            _, predicted = torch.max(logit, -1)
            _acc = predicted.eq(y).sum()
            _num = y.size(0)
            loss += (_loss * _num).item()
            acc += _acc.item()
            num += _num
            i += 1
            if not eval_full_data:
                if i == 100:
                    break
    loss /= num
    acc /= num
    return loss, acc, num


# def get_parameters(params_model, deepcopy=True):
#     ans = OrderedDict()
#     for name, params in params_model.items():
#         if deepcopy:
#             if 'weight' in name or 'bias' in name:
#                 params = params.clone().detach()
#                 ans[name] = params
#     return ans
def get_parameters(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, params in params_model.items():
        ans[name] = params.clone().detach()
    return ans


def get_buffers(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, buffers in params_model.items():
        if deepcopy:
            if 'weight' in name or 'bias' in name:
                continue
            buffers = buffers.clone().detach()
            ans[name] = buffers
    return ans

def get_cpu_param(param):
    ans = OrderedDict()
    for name, param_buffer in param.items():
        ans[name] = param_buffer.clone().detach().cpu()
    torch.cuda.empty_cache()
    return ans

def get_gpu_param(param, device=None):
    ans = OrderedDict()
    for name, param_buffer in param.items():
        ans[name] = param_buffer.clone().detach().to(device)
    return ans


class CycleDataloader:
    def __init__(self, dataloader, epoch=-1, seed=None) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.seed = seed
        self._data_iter = None
        self._init_data_iter()

    def _init_data_iter(self):
        if self.epoch == 0:
            raise StopIteration()

        if self.seed is not None:
            torch.manual_seed(self.seed + self.epoch)
        self._data_iter = iter(self.dataloader)
        self.epoch -= 1

    def __next__(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._init_data_iter()
            return next(self._data_iter)

    def __iter__(self):
        return self


###
class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            # lll = torch.abs(tensor1 - tensor2).mean(dim=(2,))
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer, y_input=None, diversity_loss_type=None):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        if diversity_loss_type == 'div2':
            y_input_dist = self.pairwise_distance(y_input, how='l1')
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        if diversity_loss_type == 'div2':
            return torch.exp(-torch.mean(noise_dist * layer_dist * torch.exp(y_input_dist)))
        else:
            return torch.exp(-torch.mean(noise_dist * layer_dist))






