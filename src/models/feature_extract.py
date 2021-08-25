import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class FeatureExtract(nn.Module):
    """
    Feature extractor mode for windowed Time-Series input.
    This module extract n features by FIR filtering.
    """

    def __init__(self, d_in: int, d_out: int = 16, ksz: int = 2, stride: int = 1):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.kernel_size = ksz
        self.stride = stride

        #self.norm = nn.GroupNorm(num_groups=self.d_in, num_channels=self.d_out * self.d_in)
        self.filter = nn.Conv1d(in_channels=1, out_channels=self.d_out, kernel_size=self.kernel_size,
                                stride=self.stride)
        self.norm = nn.LayerNorm(self.d_in * self.d_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [BS, seq_len_in, channels]
        #seq_len_in = x.size(2)
        #BS = x.size(0)

        # Channel First notation. [BS, channel, seq_len_in]
        xx = x.transpose(2, 1)

        # [BS * d_in, 1, seq_len]
        xx = torch.cat(xx.chunk(self.d_in, dim=1), dim=0)

        # [BS * d_in, d_out, seq_len_out]
        xx = self.filter(xx)

        # Reshape vector to [BS, d_out * d_in, seq_len_out]
        xx = torch.cat(xx.chunk(self.d_in, dim=0), dim=1)

        # Transpose and apply normalization [BS, seq_len_out, d_out]
        xx = self.norm(xx.transpose(2,1))

        return xx


if __name__ == '__main__':
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    a = np.array([1, 10, 100])[:, np.newaxis]
    b = np.arange(10)[:, np.newaxis]
    c = np.matmul(b, a.T)

    x = torch.from_numpy(c[np.newaxis, : ]).float()
    model = FeatureExtract(d_in=3, d_out=4, ksz=2, stride=1)
    #with torch.no_grad():
    #    model.filter.weight.fill_(1)
    #    model.filter.bias.fill_(0)

    x_out = model(x).detach().numpy()

    plt.figure()
    sns.heatmap(x[0])

    plt.figure()
    sns.heatmap(x_out[0])
    plt.show()


