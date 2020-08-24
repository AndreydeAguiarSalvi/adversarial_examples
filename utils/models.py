import math
import torch.nn as nn


def freeze_conv_and_mlp(model: nn.Module):
    for name, param in model.named_parameters():
        if 'conv' in name or 'linear' in name: param.requires_grad = False


def unfreeze_conv_and_mlp(model: nn.Module):
    for name, param in model.named_parameters():
        if 'conv' in name or 'linear' in name: param.requires_grad = True


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, has_Dropout=True, in_3_ch=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3 if in_3_ch else 1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout() if has_Dropout else Identity(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout() if has_Dropout else Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, has_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if has_batchnorm else Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if has_batchnorm else Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, has_batchnorm = True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if has_batchnorm else Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if has_batchnorm else Identity()
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4) if has_batchnorm else Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
# resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
# resnet101 = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
class ResNet(nn.Module):

    def __init__(self, kind='18', num_classes=1000, has_batchnorm=True, in_3_ch=True):
        self.inplanes = 64
        if kind == '18':
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif kind == '50':
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif kind == '101':
            block = Bottleneck
            layers = [3, 4, 23, 3]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 if in_3_ch else 1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if has_batchnorm else Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, has_batchnorm=has_batchnorm)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, has_batchnorm=has_batchnorm)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, has_batchnorm=has_batchnorm)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, has_batchnorm=has_batchnorm)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, has_batchnorm=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion) if has_batchnorm else Identity(),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, has_batchnorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, has_batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# vgg16 = VGG(make_layers(cfg['D']), **kwargs)
# vgg16_bn = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
# vgg19 = VGG(make_layers(cfg['E']), **kwargs)
# vgg19_bn = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
class VGG(nn.Module):

    def __init__(self, kind='16', num_classes=1000, has_batchnorm=True, has_dropout=True, in_3_ch=True):
        super(VGG, self).__init__()
        if kind == '16':
            if has_batchnorm: self.features = make_layers(cfg['D'], batch_norm=True, in_3_ch=in_3_ch)
            else: self.features = make_layers(cfg['D'], in_3_ch=in_3_ch)
        elif kind == '19':
            if has_batchnorm: self.features = make_layers(cfg['E'], batch_norm=True, in_3_ch=in_3_ch)
            else: self.features = make_layers(cfg['E'], in_3_ch=in_3_ch)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout() if has_dropout else Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout() if has_dropout else Identity(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, in_3_ch=True):
    layers = []
    in_channels = 3 if in_3_ch else 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x