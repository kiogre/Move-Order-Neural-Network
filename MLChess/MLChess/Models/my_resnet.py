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
    

class ChessBackbone(torchvision.models.ResNet):
    """
    This is just a piece of the ResNet, so I can add this to something else, 
    like a pointer attention mechanism or something else, like the idea in the 
    other computer to concat the otput with the output of another network
    """
    def __init__(self, layers=[2, 2, 2, 2]):
        super(ChessBackbone, self).__init__(
            block = MyBlock,
            layers = layers
        )
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)




    def forward(self, x):
        
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

class ChessDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(ChessDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 4 * 4 * 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 → 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 13, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.deconv(x)  # (batch, 13, 8, 8)


class ChessValuePolicy(nn.Module):
    def __init__(self, latent_dim=512, n_moves=1968):
        super(ChessValuePolicy, self).__init__()
        self.policy_head = nn.Linear(latent_dim, n_moves)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, z, mask):
        policy = self.policy_head(z)
        value = self.value_head(z)
        mask = mask.bool()
        policy = policy.masked_fill(mask == 0, float('-inf'))
        return policy, value


class FullChessModel(nn.Module):
    def __init__(self, latent_dim=512, n_moves=1968):
        super(FullChessModel, self).__init__()
        self.backbone = ChessBackbone()
        self.decoder = ChessDecoder(latent_dim=latent_dim)
        self.vp_head = ChessValuePolicy(latent_dim=latent_dim, n_moves=n_moves)
        self.sem_dim = latent_dim // 2  # primi 256 → siamese

    def forward_phase1(self, board_a, board_b):
        """Fase 1: siamese + autoencoder"""
        z_a = self.backbone(board_a)
        z_b = self.backbone(board_b)

        z_sem_a = z_a[:, :self.sem_dim]
        z_sem_b = z_b[:, :self.sem_dim]

        recon_a = self.decoder(z_a)
        recon_b = self.decoder(z_b)

        return z_sem_a, z_sem_b, recon_a, recon_b, board_a, board_b

    def forward_phase2(self, board, mask):
        """Fase 2: value + policy"""
        z = self.backbone(board)
        return self.vp_head(z, mask)