import torch
from torch import nn
from torch.nn.parameter import Parameter

def adaptive_kernel_size(channel, gamma=2, b=1):
    """Calculate adaptive kernel size based on channel number.

    Args:
        channel: Number of channels of the input feature map
        gamma: Hyperparameter to control the relationship between channel and kernel size
        b: Hyperparameter to control the relationship between channel and kernel size

    Returns:
        k_size: Calculated kernel size
    """
    k_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
    if k_size % 2 == 0:
        k_size += 1
    return k_size

class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=None, k_size=3):
        super(ECA, self).__init__()
        # 自适应选择卷积核大小
        self.k_size = adaptive_kernel_size(channel) if channel is not None else k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous()).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        