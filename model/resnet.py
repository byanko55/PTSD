from .nnmodel import *

__all__ = [
    "ResEncoder",
    "ResBlock",
    "ResClassifier",
    "MyResnet"
]


class ResEncoder(nn.Module):
    def __init__(
        self,
        out_planes:int = 64
    ) -> None:
        """
        First few hidden layers before the input data passes to residual net.

        Args:
            out_planes (int, optional): number of output channels.
        """

        super().__init__()

        self.conv = nn.Conv2d(3, out_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

 
class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes:int,
        out_planes:int,
        blocks:int = 2,
        stride:int = 1,
    ) -> None:
        """
        Residual learning layers. 

        Args:
            in_planes (int): number of input channels.
            out_planes (int): number of output channels.
            blocks (int, optional): define the number of residual components.
            stride (int, optional): stride of the convolution.
        """

        super().__init__()

        downsample = None
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.basic_blocks = []
        self.basic_blocks.append(
            (self.res_component(in_planes, out_planes, stride), downsample)
        )

        for _ in range(1, blocks):
            self.basic_blocks.append(
                (self.res_component(out_planes, out_planes), None)
            )

    def res_component(self, in_planes:int, out_planes:int, stride:int = 1):
        """
        A single cycle of residual feed forwarding layers.

        Args:
            in_planes (int): number of input channels.
            out_planes (int): number of output channels.
            stride (int, optional): stride of the convolution.
        """

        bb = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        return bb
    
    def cuda(self):
        for bb, downsample in self.basic_blocks:
            bb.to("cuda")

            if downsample != None:
                downsample.to("cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bb, downsample in self.basic_blocks:
            identity = x
            z = bb(x)

            if downsample != None:
                identity = downsample(x)

            z += identity
            x = self.relu(z)

        return x


class ResClassifier(nn.Module):
    def __init__(
        self,
        in_planes: int = 128,
        num_classes: int = 1000
    ) -> None:
        """
        Last few layers assigned for a classification task.

        Args:
            in_planes (int, optional): number of input channels.
            num_classes (int, optional): number of class types.
        """

        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(in_planes*4*4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

class MyResnet(nn.Module):
    def __init__(
        self,
        num_classes:int = 7
    ) -> None:
        """
        Custom Resnet consisting of two residual blocks.
        """

        super().__init__()

        self.en = ResEncoder(out_planes=64)
        self.layer1 = ResBlock(in_planes=64, out_planes=64, blocks=2)
        self.layer2 = ResBlock(in_planes=64, out_planes=128, blocks=2, stride=2)
        self.cl = ResClassifier(in_planes=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.en(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cl(x)

        return x
    
    def cuda(self):
        self.layer1.cuda()
        self.layer2.cuda()

        return super().cuda()