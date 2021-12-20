# VGG Implementation based on kuangliu's pytorch-cifar implementation
# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn

cfg = {
    'VGG11': ([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], [(512, 4096), (4096, 4096)], 4096, 32)
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, pixel_h=32, pixel_w=32):
        super(VGG, self).__init__()
        conv_cfg, fc_cfg, out_channels, pools_divisor = cfg[vgg_name]

        self.features = self._make_conv_layers(conv_cfg)

        conv_out_feature_size = int(
            pixel_w / pools_divisor * pixel_h / pools_divisor)
        fc_cfg = [(fc_cfg[0][0] * conv_out_feature_size,
                   fc_cfg[0][1])] + fc_cfg[1:]
        linear_layers = self._make_fc_layers(fc_cfg)

        linear_layers += [nn.Linear(out_channels, num_classes)]
        self.classifier = nn.Sequential(*linear_layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_conv_layers(self, cfg):
        layers = [nn.Dropout(p=0.2)]
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_fc_layers(self, cfg):
        layers = []
        for in_chan, out_chan in cfg:
            layers += [nn.Linear(in_chan, out_chan)]
            layers += [nn.ReLU(inplace=True)]
        return layers
