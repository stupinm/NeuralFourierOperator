import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1, self.n2 = n_modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, 2))

    def forward(self, x):
        n1, n2 = self.n1, self.n2
        batchsize = x.shape[0]

        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :n1, :n2] = compl_mul2d(x_ft[:, :, :n1, :n2], self.weights1)
        out_ft[:, :, -n1:, :n2] = compl_mul2d(x_ft[:, :, -n1:, :n2], self.weights2)

        out = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return out


class NeuralFourierBlock(nn.Module):
    def __init__(self, width, n_modes, activation=True):
        super(NeuralFourierBlock, self).__init__()
        self.conv = SpectralConv2d(width, width, n_modes)
        self.shortcut = nn.Conv1d(width, width, 1)
        self.bn = torch.nn.BatchNorm2d(width)
        self.activation = activation
        self.width = width

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class FourierNet(nn.Module):
    def __init__(self, n_layers, n_modes, width, t_in, t_out):
        super(FourierNet, self).__init__()
        self.n_modes = n_modes
        self.width = width
        self.fc0 = nn.Linear(t_in + 2, self.width)

        layers = [NeuralFourierBlock(width, n_modes) for i in range(n_layers - 1)]
        layers.append(NeuralFourierBlock(width, n_modes, activation=False))
        self.backbone = nn.Sequential(*layers)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, t_out)

    def forward(self, x):
        out = self.fc0(x)

        out = out.permute(0, 3, 1, 2)
        out = self.backbone(out)
        out = out.permute(0, 2, 3, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out
