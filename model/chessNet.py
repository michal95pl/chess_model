import torch.nn as nn
import torch.nn.functional as F

NUMBER_OF_MOVES = 4992
HISTORY_PLANES = 3

class ChessNet(nn.Module):

    def __init__(self, num_res_blocks, num_backbone_filters, num_policy_filters, num_value_filters):
        super().__init__()

        self.start_block = nn.Sequential(
            nn.Conv2d(13 * HISTORY_PLANES, num_backbone_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_backbone_filters),
            nn.SiLU(inplace=True)  # instead of ReLU
        )

        self.back_bone = nn.ModuleList(
            [SeResBlock(num_backbone_filters) for _ in range(num_res_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_backbone_filters, num_policy_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_policy_filters),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_policy_filters * 8 * 8, NUMBER_OF_MOVES),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_backbone_filters, num_value_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_value_filters),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_value_filters * 8 * 8, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.start_block(x)
        for block in self.back_bone:
            x = block(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy

# https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249
class SELayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # avg for each channel
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.SiLU(inplace=True), # instead of ReLU
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, num_channels) # (8x8) -> (1x1)
        y = self.excitation(y).view(batch_size, num_channels, 1, 1)
        return x * y.expand_as(x)


class SeResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.SiLU(inplace=True), # instead of ReLU
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False), # bias false because of batchnorm
            nn.BatchNorm2d(num_hidden),
            SELayer(num_hidden, 8)
        )

    def forward(self, x):
        residual = x
        x = self.path(x)
        x += residual
        x = F.silu(x, inplace=True) # instead of ReLU
        return x

# import visualtorch
import matplotlib.pyplot as plt
# device = torch.device("cpu")
# model = ChessNet(80, 3)

# img = visualtorch.graph_view(
#     model,
#     input_shape=(1, 13, 8, 8),
#     to_file="chessNet_graph.png"
# )

# img = visualtorch.layered_view(
#     model,
#     input_shape=(1, 13, 8, 8)
# )

# plt.axis('off')
# plt
# plt.imshow(img)
# plt.show()