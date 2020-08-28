import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, x=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


valid_layers = {
    'Dropout': nn.Dropout,
    'BatchNorm': nn.BatchNorm2d,
    'InstanceNorm': nn.InstanceNorm2d,
    'Identity': Identity
}


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class AlexNet(nn.Module):

    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1 if args['dataset'] == 'MNIST' else 3, 64, kernel_size=11, stride=4, padding=2),
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
            valid_layers[args['dropout']](),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            valid_layers[args['dropout']](),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(args['classes'])),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()
        self.features = self._make_layers(args)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            valid_layers[args['dropout']](),
            nn.Linear(512, 256),
            nn.ReLU(True),
            valid_layers[args['dropout']](),
            nn.Linear(256, len(args['classes'])),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, args):
        layers = []
        in_channels = 1 if args['dataset'] == 'MNIST' else 3
        for x in cfg[args['model']]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           valid_layers[args['batchnorm']](x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, args, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = valid_layers[args['batchnorm']](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = valid_layers[args['batchnorm']](planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                valid_layers[args['batchnorm']](self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, args, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = valid_layers[args['batchnorm']](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = valid_layers[args['batchnorm']](planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = valid_layers[args['batchnorm']](self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                valid_layers[args['batchnorm']](self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1 if args['dataset'] == 'MNIST' else 3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        block = BasicBlock if '18' in args['model'] or '34' in args['model'] else Bottleneck
        if '18' in args['model']: num_blocks = [2, 2, 2, 2]
        elif '34' in args['model']: num_blocks = [3, 4, 6, 3]
        elif '50' in args['model']: num_blocks = [3, 4, 6, 3]
        elif '101' in args['model']: num_blocks = [3, 4, 23, 3]
        elif '152' in args['model']: num_blocks = [3, 8, 36, 3]

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], args, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], args, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], args, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], args, stride=2)
        self.linear = nn.Linear(512*block.expansion, len(args['classes']))

    def _make_layer(self, block, planes, num_blocks, args, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, args, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out