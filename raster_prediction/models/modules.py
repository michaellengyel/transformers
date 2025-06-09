import math
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1))
        self.down_2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = nn.functional.relu(self.down_1(x))
        x = nn.functional.relu(self.down_2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        )
        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = nn.functional.relu(self.up_1(x))
        x = nn.functional.sigmoid(self.up_2(x))
        return x


if __name__ == '__main__':

    model = Decoder(64, 3)

    # x = torch.zeros(8, 3, 64, 64)
    x = torch.zeros(8, 64, 16, 16)
    yp = model(x)
    print(yp.shape)
