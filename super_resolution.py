import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from pathlib import Path

class SuperResDataset(Dataset):
    def __init__(self, path):
        assert Path(path).exists()
        self.root_dir = path
        self.high_res_ds = ImageFolder(path, transform=transforms.Compose([
                                                   transforms.Resize((128, 128)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
        self.low_res_ds = ImageFolder(path, transform=transforms.Compose([
                                                   transforms.Resize((64, 64)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
        assert len(self.high_res_ds) == len(self.low_res_ds)
    
    def __getitem__(self, idx):
        high_res = self.high_res_ds.__getitem__(idx)[0]
        low_res = self.low_res_ds.__getitem__(idx)[0]

        return (low_res, high_res)

    def __len__(self):
        return len(self.high_res_ds)

class SuperResModel(nn.Module):

    def __init__(self):
        super(SuperResModel, self).__init__()

        number_f = 32
        self.subpixel_convolutional_block = SubPixelConvolutionalBlock(kernel_size=3, n_channels=3, scaling_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels=3,            out_channels=number_f,   kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(number_f)
        self.conv2 = nn.Conv2d(in_channels=number_f,     out_channels=number_f,   kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(number_f)
        self.conv3 = nn.Conv2d(in_channels=number_f,     out_channels=number_f*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(number_f*2)
        self.conv4 = nn.Conv2d(in_channels=number_f*2,   out_channels=number_f*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(number_f*2)
        self.conv5 = nn.Conv2d(in_channels=number_f*2 , out_channels=number_f*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(number_f*4)
        self.conv6 = nn.Conv2d(in_channels=number_f*4 , out_channels=3,          kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh =  nn.Tanh()
        
    def forward(self, X):
        bicubic_upsample = self.upsample(X)
        x1 = self.subpixel_convolutional_block(X)

        x2 = F.relu(self.conv1(x1))
        x2 = self.bn1(x2)

        x2 = F.relu(self.conv2(x2))
        x2 = self.bn2(x2)

        x2 = F.relu(self.conv3(x2))
        x2 = self.bn3(x2)

        x2 = F.relu(self.conv4(x2))
        x2 = self.bn4(x2)

        x2 = F.relu(self.conv5(x2))
        x2 = self.bn5(x2)

        x2 = self.conv6(x2)

        x3 = self.tanh(x2 + bicubic_upsample)
        return x3

class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output