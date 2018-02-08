import numpy as np
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(model, method='kaiming'):
    for m in model.modules():
        print m
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            if method is 'kaiming':
                init.kaiming_normal(m.weight.data, np.sqrt(2.0))
            elif method is 'xavier':
                init.xavier_normal(m.weight.data, np.sqrt(2.0))
            elif method is 'normal':
                init.normal(m.weight.data, mean=0, std=0.02)

            if m.bias is not None:
                init.constant(m.bias.data, 0)
