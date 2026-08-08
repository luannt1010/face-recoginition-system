import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection=False):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.shortcut = self.make_shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def make_shortcut(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.skip_connection:
            x += self.shortcut(identity)
        x = self.relu(x)
        x = self.max_pool(x)
        return x 


class SimpleResNet(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.4):
        super().__init__()
        self.block1 = CNNBlock(in_channels=3, out_channels=32, skip_connection=True)
        self.block2 = CNNBlock(in_channels=32, out_channels=64, skip_connection=True)
        self.block3 = CNNBlock(in_channels=64, out_channels=128, skip_connection=True)
        self.block4 = CNNBlock(in_channels=128, out_channels=256, skip_connection=True)
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.extract_embed = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(256*7*7, 2048), nn.ReLU(), nn.Dropout(self.dropout),
                                        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(self.dropout),
                                        nn.Linear(1024, self.embedding_dim))
    
    def forward(self, x):
        for block in [self.block1, self.block2, self.block3, self.block4]:
            x = block(x)
        x = self.extract_embed(x)
        return x



