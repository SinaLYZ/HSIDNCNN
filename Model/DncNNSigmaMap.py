import torch
import torch.nn as nn


class DnCNNSigmaMap(nn.Module):
    """
    DnCNN for sigma-conditioned image denoising.

    This version takes a multi-channel input such as:
        [noisy_image, sigma_map]

    and predicts only the noise residual of the image channel.

    Final clean image can be obtained by:
        clean = noisy_image - model(model_input)

    Args:
        in_channels (int): Number of input channels.
                           For sigma-conditioned grayscale input, use 2.
        out_channels (int): Number of output channels.
                            For residual prediction of one grayscale image, use 1.
        depth (int): Total number of convolutional layers.
        num_features (int): Number of feature maps in hidden layers.
        kernel_size (int): Convolution kernel size.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
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
        # Important:
        # output must be 1 channel for predicted image noise,
        # not equal to in_channels.
        layers.append(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted noise residual for the image channel.
        Input x is expected to have shape [B, in_channels, H, W].
        For sigma-conditioned grayscale denoising:
            x[:, 0:1, :, :] = noisy image
            x[:, 1:2, :, :] = sigma map
        """
        return self.network(x)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience function: returns denoised image directly.

        Uses only the first channel of x as the noisy image.
        Example:
            x[:, 0:1, :, :] = noisy image
            x[:, 1:2, :, :] = sigma map
        """
        noise = self.forward(x)
        noisy_image = x[:, 0:1, :, :]
        clean = noisy_image - noise
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
    # Example test for sigma-conditioned input
    model = DnCNNSigmaMap(in_channels=2, out_channels=1, depth=17, num_features=64)

    x = torch.randn(4, 2, 64, 64)  # batch of 4 samples: [noisy_image, sigma_map]
    predicted_noise = model(x)
    denoised_image = model.denoise(x)

    print("Input shape:", x.shape)
    print("Predicted noise shape:", predicted_noise.shape)
    print("Denoised image shape:", denoised_image.shape)