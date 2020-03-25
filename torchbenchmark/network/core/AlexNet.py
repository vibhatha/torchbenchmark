import torch.nn as nn

from torchbenchmark.network.util.FlattenCustom import FlattenCustom


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2), )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2), )

        self.layer3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True), )

        self.layer4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True), )

        self.layer5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2), )

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)))

        self.flatten = nn.Sequential(FlattenCustom())

        self.layer6 = nn.Sequential(nn.Dropout(),
                                    nn.Linear(256 * 6 * 6, 4096),
                                    nn.ReLU(inplace=True), )

        self.layer7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.layer8 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
