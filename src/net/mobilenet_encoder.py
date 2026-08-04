import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, t, n_in, n_out, stride):
        super().__init__()

        n_expanded = n_in * t
        self.n_in = n_in
        self.n_out = n_out
        self.stride = stride
        self.pointwise_expand = nn.Sequential(nn.Conv2d(n_in, n_expanded, 1, 1, bias=False),
                                              nn.BatchNorm2d(n_expanded), nn.PReLU(n_expanded))
        self.depthwise = nn.Sequential(nn.Conv2d(n_expanded, n_expanded, 3, stride=stride, padding=1, groups=n_expanded, bias=False),
                                       nn.BatchNorm2d(n_expanded), nn.PReLU(n_expanded))
        self.pointwise_down = nn.Sequential(nn.Conv2d(n_expanded, n_out, 1, 1, bias=False),
                                            nn.BatchNorm2d(n_out))

    def forward(self, x):
        identity = x
        output = self.pointwise_down(self.depthwise(self.pointwise_expand(x)))
        if self.stride == 1 and identity.shape[1] == self.n_out:
            output = output + identity
        return output

class MakeBlockBottleNeck(nn.Module):
    def __init__(self, t, c_in, c_out, n, s):
        super().__init__()
        self.t = t
        self.n = n
        self.s = s
        self.c_in = c_in
        self.c_out = c_out
        self.blocks = self._make_blocks()

    def _make_blocks(self):
        blocks = [BottleNeck(self.t, self.c_in, self.c_out, self.s)]
        for _ in range(self.n-1):
            blocks.append(BottleNeck(self.t, self.c_out, self.c_out, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.conv3x3 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                                     nn.BatchNorm2d(64), nn.PReLU(64))
        self.depthwise = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
                                       nn.BatchNorm2d(64), nn.PReLU(64))
        self.block1 = MakeBlockBottleNeck(2, 64, 64, 5, 2)
        self.block2 = MakeBlockBottleNeck(4, 64, 128, 1, 2)
        self.block3 = MakeBlockBottleNeck(2, 128, 128, 6, 1)
        self.block4 = MakeBlockBottleNeck(4, 128, 128, 1, 2)
        self.block5 = MakeBlockBottleNeck(2, 128, 128, 2, 1)
        self.conv1x1 = nn.Sequential(nn.Conv2d(128, 512, 1, 1, bias=False),
                                     nn.BatchNorm2d(512), nn.PReLU(512))
        self.linear_gdconv = nn.Sequential(nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
                                                nn.BatchNorm2d(512))
        self.linear_conv1x1 = nn.Sequential(nn.Conv2d(512, embedding_dim, 1, 1, bias=False),
                                            nn.BatchNorm2d(embedding_dim), nn.Flatten())

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.depthwise(x)
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            x = block(x)
        x = self.conv1x1(x)
        x = self.linear_gdconv(x)
        output = self.linear_conv1x1(x)
        return output



