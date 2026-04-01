import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN for image denoising.

    This version predicts the noise residual.
    Final clean image can be obtained by:
        clean = noisy - model(noisy)

    Args:
        in_channels (int): Number of input image channels.
                           1 for grayscale, 3 for RGB.
        depth (int): Total number of convolutional layers.
        num_features (int): Number of feature maps in hidden layers.
        kernel_size (int): Convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 17,
        num_features: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if depth < 3:
            raise ValueError("DnCNN depth must be at least 3.")

        padding = kernel_size // 2
        layers = []

        # First layer: Conv + ReLU
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=num_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv
        layers.append(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted noise residual.
        """
        return self.network(x)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience function: returns denoised image directly.
        """
        noise = self.forward(x)
        clean = x - noise
        return clean

    def _initialize_weights(self) -> None:
        """
        Kaiming initialization, commonly used for DnCNN.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Example test
    model = DnCNN(in_channels=1, depth=17, num_features=64)

    x = torch.randn(4, 1, 64, 64)  # batch of 4 grayscale images
    predicted_noise = model(x)
    denoised_image = model.denoise(x)

    print("Input shape:", x.shape)
    print("Predicted noise shape:", predicted_noise.shape)
    print("Denoised image shape:", denoised_image.shape)