import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock2D(nn.Module):
    """Standard 2D ResNet BasicBlock."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck2D(nn.Module):
    """Standard 2D ResNet Bottleneck block."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


def create_fcs(input_size: int, output_size: int) -> nn.Sequential:
    """
    Create a small MLP that gradually halves feature dimensions
    until reaching `output_size`.
    """
    layer_sizes = [input_size, int(input_size / 2)]

    while layer_sizes[-1] > output_size:
        next_size = max(layer_sizes[-1] // 2, output_size)
        if next_size == layer_sizes[-1]:
            break
        layer_sizes.append(int(next_size))

    if layer_sizes[-1] != output_size:
        layer_sizes.append(output_size)

    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class GridFeatureResNet(nn.Module):
    """
    ResNet-based classifier operating on a constructed 2D grid
    from CBCT and HIST feature vectors.

    The grid is formed from weighted CBCT/HIST feature vectors:
        cbct_feature: [B, 512]
        hist_feature: [B, 512]
        w1, w2:       [B]

    → matrix: [B, 512, 512] → CNN → classifier.
    """

    def __init__(
        self,
        block: type[nn.Module],
        num_blocks: list[int],
        num_classes: int = 1000,
        channel: int = 1,
        include_top: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(
            channel,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = create_fcs(512 * block.expansion, num_classes)

        # He (Kaiming) init for conv, standard init for BN.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(
        self,
        block: type[nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self,
        cbct_feature: torch.Tensor,
        hist_feature: torch.Tensor | None = None,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        cbct_feature: [B, 1, 512] or [B, 512]
        hist_feature: [B, 1, 512] or [B, 512] (optional)
        w1, w2:       [B] (per-sample weights)
        """
        cbct_feature = cbct_feature.squeeze(dim=1)  # [B, 512]
        batch_size = cbct_feature.shape[0]

        if hist_feature is None:
            hist_feature = torch.ones(batch_size, 512, device=cbct_feature.device)
        else:
            hist_feature = hist_feature.squeeze(dim=1)  # [B, 512]

        # w1, w2: [B] → [B, 1, 1]
        w1 = w1.unsqueeze(1).unsqueeze(2)
        w2 = w2.unsqueeze(1).unsqueeze(2)

        # cbct_weighted: [B, 1, 512]
        cbct_weighted = w1 * cbct_feature.unsqueeze(1)
        # hist_weighted: [B, 512, 1]
        hist_weighted = w2 * hist_feature.unsqueeze(2)

        # matrix: [B, 512, 512]
        matrix = cbct_weighted + hist_weighted
        x = matrix.unsqueeze(1)  # [B, 1, 512, 512]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
