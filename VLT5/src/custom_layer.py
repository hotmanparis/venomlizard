import torch
import torch.nn as nn


# Resnet Blocks
class EmbeddingNet(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class ConditionTextImage(nn.Module):
    ''' Condition text tokens with global vision embedding

    Input:
        L (tensor): text embeddings (Batch size x seq length x hdim)
        V (tensor): global vision embedding (Batch size x hdim)
    Output:
        L (tensor) : conditioned text embeddings (Batch size x seq length x hdim)
    '''

    def __init__(self, hdim = 768, factor=2):
        super().__init__()
        # Submodules
        self.fc_0 = nn.Linear(hdim, hdim//factor)
        self.fc_1 = nn.Linear(hdim//factor, hdim)
        self.actvn = nn.ReLU()

    def forward(self, L, V):
        V = V.unsqueeze(1)
        VL = L + V
        VL = self.fc_0(self.actvn(VL))
        VL = self.fc_1(self.actvn(VL))

        L = L + VL

        return L
        

class ConditionTextImage2(nn.Module):
    ''' Condition text tokens with global vision embedding

    Input:
        L (tensor): text embeddings (Batch size x seq length x hdim)
        V (tensor): global vision embedding (Batch size x hdim)
    Output:
        L (tensor) : conditioned text embeddings (Batch size x seq length x hdim)
    '''

    def __init__(self, hdim = 768, factor=1):
        super().__init__()
        # Submodules
        self.fc_0 = nn.Linear(hdim, hdim//factor)
        self.fc_1 = nn.Linear(hdim//factor, hdim)
        self.actvn = nn.ReLU()

    def forward(self, L, V, mask):
        L = L[mask]
        mask_sum = torch.sum(mask, dim=1)

        V = torch.repeat_interleave(V, mask_sum, dim=0)

        VL = L + V
        VL = self.fc_0(self.actvn(VL))
        VL = self.fc_1(self.actvn(VL))

        return VL