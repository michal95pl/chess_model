import torch.nn as nn
import torch.nn.functional as F
import torch


class ChessNet(nn.Module):

    def __init__(self, num_hidden, num_resBlocks):
        super().__init__()

        self.device = device
        self.start_block = nn.Sequential(
            nn.Conv2d(13, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.back_bone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4992)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        for block in self.back_bone:
            x = block(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

import visualtorch
import matplotlib.pyplot as plt
device = torch.device("cpu")
model = ChessNet(80, 3)

img = visualtorch.graph_view(
    model,
    input_shape=(1, 13, 8, 8),
    to_file="chessNet_graph.png"
)

# img = visualtorch.layered_view(
#     model,
#     input_shape=(1, 13, 8, 8)
# )

# plt.axis('off')
# plt
# plt.imshow(img)
# plt.show()