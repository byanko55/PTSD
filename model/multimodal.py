from .nnmodel import *


class MultiModalNN(nn.Module):
    def __init__(
        self,
        num_classes=7
    ) -> None:
        """
        Naive late fusion model targetting multi-modal learning tasks.
        Currently applicable for Audio + Text.

        Args:
            num_classes (int): number of class types.
        """

        super().__init__()

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(3072, 768),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, num_classes)
        )

    def forward(self, x_mel: torch.Tensor, x_tok: torch.Tensor) -> torch.Tensor:
        z_mel = self.audio_encoder(x_mel)
        #z_mel = torch.flatten(z_mel, 1)

        z = torch.cat((z_mel, x_tok), 1)
        out = self.classifier(z)

        return out