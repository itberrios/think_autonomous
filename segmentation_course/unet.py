import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms


def encoder_block(in_channels, out_channels, dropout=True):
    ''' Builds the U-net encoder block '''
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
        ]

    if dropout:
        layers.append(nn.Dropout(p=0.25))

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


def decoder_block(in_channels, middle_channels, out_channels):
    ''' builds the U-net decoder block '''
    layers = [
        nn.Conv2d(in_channels, middle_channels, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(middle_channels),
        nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(middle_channels),
        nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        ]
        
    return nn.Sequential(*layers)

    
def crop(x, encode_skip):
    ''' crops the encoder skip connection features so they fit with the decoder features
        Inputs: 
            x - decoder features
            encode_skip - encoder skip connection features
    '''
    _, _, h, w, = x.shape
    return transforms.CenterCrop((h, w))(encode_skip)



class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # encoder layers
        self.encode1 = encoder_block(3, 64)
        self.encode2 = encoder_block(64, 128)
        self.encode3 = encoder_block(128, 256)
        self.encode4 = encoder_block(256, 512)

        # base/bottom layer
        self.base = decoder_block(512, 1024, 512)

        # decoder layers
        self.decode1 = decoder_block(1024, 512, 256)
        self.decode2 = decoder_block(512, 256, 128)
        self.decode3 = decoder_block(256, 128, 64)
        self.decode4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64))

        # 1x1 convolution for the output layer
        self.output_layer = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x):
        # perform encoding
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)

        # base/bottom layer
        b = self.base(e4)

        # TEMP
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(e4.shape)
        # print(b.shape)
        # print()

        # perform decoding with skip connections from encoding layers
        d1 = self.decode1(torch.cat((crop(b, e4), b), dim=1))
        d2 = self.decode2(torch.cat((crop(d1, e3), d1), dim=1))
        d3 = self.decode3(torch.cat((crop(d2, e2), d2), dim=1))
        d4 = self.decode4(torch.cat((crop(d3, e1), d3), dim=1))

        # TEMP
        # print(d1.shape)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)

        # get final layer with correct number of classes/channels
        out = self.output_layer(d4)

        return out
        