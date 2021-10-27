""" Unet implementation adaped from github.com/constantinpape/MIPNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """ Upsample the input and change the number of channels
    via 1x1 Convolution if a different number of input/output channels is specified.
    """

    def __init__(self, scale_factor, mode='nearest',
                 in_channels=None, out_channels=None,
                 align_corners=False):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv = None

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor,
                          mode=self.mode, align_corners=self.align_corners)
        if self.conv is not None:
            return self.conv(x)
        else:
            return x

class UNetBase(nn.Module):
    """ UNet Base class implementation
    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
        return conv block for a U-Net level
    - _pooler(level)
        return pooling operation used for downsampling in-between encoders
    - _upsampler(in_channels, out_channels, level)
        return upsampling operation used for upsampling in-between decoders
    - _out_conv(in_channels, out_channels)
        return output conv layer
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      depth: depth of the network
      initial_features: number of features after first convolution
      gain: growth factor of features
      pad_convs: whether to use padded convolutions
    """

    def __init__(self, in_channels, out_channels, depth=4,
                 initial_features=64, gain=2, pad_convs=False):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = pad_convs

        # modules of the encoder path
        n_features = [in_channels] + [initial_features * gain ** level
                                      for level in range(self.depth)]
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder')
                                      for level in range(self.depth)])

        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1], part='base', level=0)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       self.depth - level - 1, part='decoder')
                                      for level in range(self.depth)])

        # the pooling layers;
        self.poolers = nn.ModuleList([self._pooler(level) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1)
                                         for level in range(self.depth)])

        if self.out_channels is not None:
            self.out_conv = self._out_conv(n_features[-1], out_channels)
        else:
            self.out_conv = None
            self.out_channels = n_features[-1]

    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        # get the difference between the shapes
        shape_diff = tuple((ish - csh) // 2
                           for ish, csh in zip(input_shape, shape_to_crop))
        if all(sd == 0 for sd in shape_diff):
            return input_
        # calculate the crop
        crop = tuple(slice(sd, sh - sd)
                     for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input):
        x = input

        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x,
                                                          encoder_out[level]))

        if self.out_conv is not None:
            x = self.out_conv(x)
        return x


#
# 2D U-Net implementations
class UNet2d(UNetBase):
    """ 2d U-Net for segmentation as described in
    https://arxiv.org/abs/1505.04597
    """
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        padding = 1 if self.pad_convs else 0
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       kernel_size=3, padding=padding),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels,
                                       kernel_size=3, padding=padding),
                             nn.ReLU())

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels, level):
        # use bilinear upsampling + 1x1 convolutions
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2, mode='bilinear')

    # pooling via maxpool2d
    def _pooler(self, level):
        return nn.MaxPool2d(2)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1)

