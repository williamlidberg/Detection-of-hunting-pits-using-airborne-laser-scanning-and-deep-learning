import torch
import torch.nn as nn

import math

from ..layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from ..layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from ..layers.channel_attention_layer import SE_Conv_Block
from ..layers.scale_attention_layer import scale_atten_convblock
from ..layers.nonlocal_layer import NONLocalBlock2D


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, depth=4, feature_scale=4,
                 is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        assert type(depth) is int, 'Depth needs to be an integer'
        assert depth >= 3, f'Depth needs to be larger than two - is: {depth}'
        max_depth = math.log2(self.out_size[0])
        max_depth = max(max_depth, math.log2(self.out_size[1]))
        assert depth < max_depth, f'Depth: {depth} too large for input shape'

        # configure number of filter entries and output sizes based on
        # configured depth (one more than depth)
        filters = [2**f for f in range(6, 6 + depth + 1)]
        filters = [int(x / self.feature_scale) for x in filters]
        out_sizes = [(math.floor(self.out_size[0]/2**i),
                      math.floor(self.out_size[1]/2**i)) for i in range(depth)]

        # downsampling
        self.conv_layers = nn.ModuleList()
        self.maxpool_layers = nn.ModuleList()
        # iterate over number of input channels + all filter sizes except the
        # last one
        for i, filt in enumerate([self.in_channels, *filters[:-2]]):
            # configure dropout for last considered conv layer
            drop = True if i == len(filters) - 2 else False
            self.conv_layers.append(conv_block(filt, filters[i],
                                               drop_out=drop))
            self.maxpool_layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.center = conv_block(filters[-2], filters[-1], drop_out=True)

        # attention blocks
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[-1],
                                           inter_channels=filters[-1] // 4)

        self.attentionblocks = nn.ModuleList()
        for i in range(len(filters) - 2, 1, -1):
            block = MultiAttentionBlock(in_size=filters[i-1],
                                        gate_size=filters[i],
                                        inter_size=filters[i-1],
                                        nonlocal_mode=nonlocal_mode,
                                        sub_sample_factor=attention_dsample)
            self.attentionblocks.append(block)

        # upsampling
        self.up_concats = nn.ModuleList()
        self.ups = nn.ModuleList()
        # add one upsampling layer for each downsampling layer -> there are
        # depth many out_sizes, but depth+1 many filter sizes
        for i, shape in enumerate(out_sizes[::-1]):
            self.up_concats.append(UpCat(filters[depth-i], filters[depth-i-1],
                                         self.is_deconv))
            drop = True if i == 0 else False
            self.ups.append(SE_Conv_Block(filters[depth-i], filters[depth-i-1],
                                          shape, drop_out=drop))

        # deep supervision
        self.dsvs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.dsvs.append(UnetDsv3(in_size=filters[i], out_size=4,
                                      scale_factor=self.out_size))
        self.dsvs.append(nn.Conv2d(in_channels=filters[0], out_channels=4,
                                   kernel_size=1))

        self.scale_att = scale_atten_convblock(in_size=4*depth, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        convs = []
        ups = []

        # Feature Extraction
        x = inputs
        for conv, maxpool in zip(self.conv_layers, self.maxpool_layers):
            convs.append(conv(x))
            x = maxpool(convs[-1])

        # Gating Signal Generation
        x = self.center(x)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        prev = convs.pop()
        for i, (up_concat, up) in enumerate(zip(self.up_concats, self.ups)):
            # perform up covolution
            x = up_concat(prev, x)
            # apply non local layer only at lowest level
            if i == 0:
                x = self.nonlocal4_2(x)
            # apply channel attention
            # for now we ignore the returned attention weights
            x, _ = up(x)
            if i < len(self.attentionblocks):
                # apply spatial attention
                # for now we ignore the returned attention weights
                prev, _ = self.attentionblocks[i](convs.pop(), x)
            # at the last loop iteration, we do not need to setup the next
            # iteration
            elif len(convs) != 0:
                prev = convs.pop()
            ups.append(x)

        # Deep Supervision
        dsvs = [f(x) for f, x in zip(self.dsvs, ups)]
        dsv_cat = torch.cat(dsvs, dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out
