import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MyBlock(nn.Module):
    expansion = 1  # non serve 2 se non stai cambiando canali

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(MyBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    

class MyResNet(torchvision.models.ResNet):
    """
    Questo sarà la prima prova del mio modello, una rete convoluzionale
    La rete convoluzionale tecnicamente dovrebbe essere in grado di capire
    dei pattern spaziali 
    """
    def __init__(self, layers=[2, 2, 2, 2]):
        super(MyResNet, self).__init__(
            block = MyBlock,
            layers = layers
        )
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)

        self.fc = nn.Linear(512, 1968)

        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # valore tra -1 e 1
        )

    def forward(self, x, mask):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Ora attraverso i layer della ResNet
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        x = self.layer4(x)
        
        # Il flatten che causa il problema
        x = torch.flatten(x, start_dim=1)
        
        # Calcoli che hai fatto
        policy = self.fc(x)
        value = self.value_head(x)
        
        mask = mask.clone().detach().bool()
        policy = policy.masked_fill(mask == 0, float("-inf"))
        return policy, value
    

class NotUntilEndResNet(torchvision.models.ResNet):
    """
    This is just a piece of the ResNet, so I can add this to something else, 
    like a pointer attention mechanism or something else, like the idea in the 
    other computer to concat the otput with the output of another network
    """
    def __init__(self, layers=[2, 2, 2, 2]):
        super(NotUntilEndResNet, self).__init__(
            block = MyBlock,
            layers = layers
        )
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)




    def forward(self, x, mask):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Ora attraverso i layer della ResNet
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        x = self.layer4(x)
        
        # Il flatten che causa il problema
        x = torch.flatten(x, start_dim=1)
        
        return x
